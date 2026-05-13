# Week 5: Production Build Systems, Testing & Tooling

## 1. Modern CMake Philosophy: Targets, Not Variables

The single most important shift in modern CMake (3.0+) is thinking in **targets** instead of
global variables. Legacy CMake used `include_directories()`, `add_definitions()`, and
`link_libraries()` — all of which pollute the global scope. Modern CMake attaches properties
to targets, and those properties propagate through the dependency graph.

### PRIVATE / PUBLIC / INTERFACE

These keywords control **transitive propagation** of compile flags, include paths, and
link dependencies:

| Keyword     | Used by target itself? | Propagated to dependents? |
|-------------|:---------------------:|:-------------------------:|
| PRIVATE     | Yes                   | No                        |
| PUBLIC      | Yes                   | Yes                       |
| INTERFACE   | No                    | Yes                       |

```cmake
# rt_core is a library. Its headers live in include/
add_library(rt_core src/cyclic_executive.cpp)

# PUBLIC: dependents also need these headers (they #include them)
target_include_directories(rt_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# PRIVATE: only rt_core itself needs pthreads at link time
target_link_libraries(rt_core PRIVATE Threads::Threads)

# INTERFACE: rt_core is header-only for template parts — consumers
# need C++20 but rt_core.cpp doesn't use C++20 features itself
target_compile_features(rt_core INTERFACE cxx_std_20)
```

The generator expressions `$<BUILD_INTERFACE:...>` and `$<INSTALL_INTERFACE:...>` let the
same target work both when building in-tree and after `make install`.

### Why This Matters for Embedded/Safety-Critical

In an OKS-style project with `rt_core`, `navigation`, `serial_datalink`, etc., each library
declares its own requirements. When you `target_link_libraries(navigation PUBLIC rt_core)`,
CMake automatically propagates rt_core's PUBLIC include paths and compile flags to navigation
and anything that depends on navigation. No manual tracking.

---

## 2. CMakePresets.json — Reproducible Configurations

CMakePresets.json (CMake 3.19+) replaces the old `cmake -DCMAKE_BUILD_TYPE=Debug` dance.
It defines named configurations that anyone can use:

```
cmake --preset debug        # Debug build
cmake --preset asan         # AddressSanitizer
cmake --preset tsan         # ThreadSanitizer
cmake --build --preset debug
```

Key preset types:
- **configurePresets**: set generator, build dir, cache variables, environment
- **buildPresets**: set build parallelism, targets
- **testPresets**: set test filters, timeout, output format

A typical safety-critical project has at minimum:

| Preset  | Purpose                         | Key Flags                                 |
|---------|---------------------------------|-------------------------------------------|
| debug   | Development                     | `-O0 -g -DDEBUG`                          |
| release | Production                      | `-O2 -DNDEBUG`                            |
| asan    | Memory errors                   | `-fsanitize=address,undefined`            |
| tsan    | Data races                      | `-fsanitize=thread`                       |
| ubsan   | Undefined behavior              | `-fsanitize=undefined -fno-sanitize-recover` |
| fuzz    | Coverage-guided fuzzing         | `-fsanitize=fuzzer,address`               |
| arm     | Cross-compile for target HW     | Toolchain file + `CMAKE_SYSTEM_NAME`      |

**Critical**: ASan and TSan are **mutually exclusive** — they instrument memory differently.
You must run them as separate CI jobs.

---

## 3. GoogleTest & GoogleMock

### Test Macros

```cpp
TEST(SuiteName, TestName) {           // standalone test
    EXPECT_EQ(1 + 1, 2);             // non-fatal: continues on failure
    ASSERT_NE(ptr, nullptr);          // fatal: aborts this test
}

TEST_F(FixtureClass, TestName) {      // uses SetUp()/TearDown()
    EXPECT_TRUE(fixture_member_.isValid());
}
```

`EXPECT_*` vs `ASSERT_*`: Use EXPECT by default (reports all failures). Use ASSERT only
when continuing would crash (e.g., null pointer you're about to deref).

### Parameterized Tests

```cpp
class QueueSizeTest : public ::testing::TestWithParam<size_t> {};

TEST_P(QueueSizeTest, PushPopRoundTrip) {
    SpscQueue<int> q(GetParam());
    q.push(42);
    EXPECT_EQ(q.pop(), 42);
}

INSTANTIATE_TEST_SUITE_P(Sizes, QueueSizeTest,
    ::testing::Values(1, 4, 64, 1024, 65536));
```

### GoogleMock

Mock objects verify **interactions** — did function X get called with argument Y?

```cpp
class MockSensor : public ISensor {
public:
    MOCK_METHOD(float, read, (), (override));
    MOCK_METHOD(bool, calibrate, (int level), (override));
};

TEST(Controller, ReadsFromSensor) {
    MockSensor sensor;
    EXPECT_CALL(sensor, read()).Times(1).WillOnce(Return(3.14f));
    Controller ctrl(&sensor);
    ctrl.update();
}
```

### Death Tests

Verify that assertion failures / aborts actually fire:

```cpp
TEST(QueueDeathTest, PopFromEmpty) {
    SpscQueue<int> q(4);
    EXPECT_DEATH(q.pop(), "Queue is empty");  // regex match on stderr
}
```

---

## 4. Fuzzing with libFuzzer

### How Coverage-Guided Fuzzing Works

1. Compile with `-fsanitize=fuzzer,address` — clang instruments every branch edge
2. Fuzzer generates random byte sequences, feeds them to `LLVMFuzzerTestOneInput`
3. After each run, it checks which **new code paths** were reached
4. Inputs that trigger new coverage are added to the **corpus**
5. Corpus grows over time, exploring deeper code paths
6. Crashes (ASan violations, asserts, timeouts) are saved as reproducer files

### Writing a Harness

```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;  // need minimum header
    ParseMessage({data, size});
    return 0;
}
```

**Key rules**:
- Return 0 (non-zero = "skip this input" in newer libFuzzer)
- Don't call `exit()` — fuzzer runs thousands of iterations per second in-process
- Handle inputs smaller than your minimum gracefully (return 0)
- Don't use global mutable state (fuzzer is single-threaded but rapid)

### Corpus Management

```bash
mkdir corpus
# Run fuzzer, writing new corpus entries:
./fuzz_parser corpus/ -max_total_time=300

# Minimize corpus (remove redundant inputs):
./fuzz_parser -merge=1 corpus_minimized/ corpus/

# Reproduce a crash:
./fuzz_parser crash-abc123def
```

### What Fuzzing Finds

- Buffer overflows (reading past input end)
- Integer overflows in length calculations
- Null derefs from unchecked parse results
- Infinite loops / excessive memory from crafted inputs
- Format string vulnerabilities

---

## 5. Profiling

### perf (Linux)

```bash
# Record 10 seconds of a running process:
perf record -g ./my_program       # -g = call graph (frame pointers)
perf report                        # interactive TUI

# Generate flame graph (requires brendangregg/FlameGraph):
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

A **flame graph** shows call stacks on the X-axis (width = time spent) and call depth on
the Y-axis. Wide bars at the top are your hot functions.

### Valgrind Callgrind (Instruction-Level Profiling)

```bash
valgrind --tool=callgrind ./my_program
callgrind_annotate callgrind.out.*    # text report
kcachegrind callgrind.out.*           # GUI visualization
```

Callgrind counts every instruction, so it's ~20-50× slower but deterministic (no sampling
noise). Good for comparing before/after.

### Valgrind Cachegrind (Cache Analysis)

```bash
valgrind --tool=cachegrind ./my_program
cg_annotate cachegrind.out.*
```

Reports L1/LL cache miss rates per function. When you see a function with 40% L1 miss rate,
that's your cache-unfriendly access pattern.

---

## 6. Code Coverage with gcov/lcov

### How It Works

1. Compile with `--coverage` (adds `-fprofile-arcs -ftest-coverage`)
2. Run your tests — this generates `.gcda` files (execution counts per branch)
3. `gcov` reads `.gcda` + `.gcno` (program structure) → per-file reports
4. `lcov` aggregates into HTML reports

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage" ..
make && ctest
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/test/*' -o filtered.info
genhtml filtered.info --output-directory coverage_html
```

### What 80% Coverage Actually Means

80% **line coverage** means 80% of executable lines were hit by at least one test. But:

- **Line coverage ≠ branch coverage**: `if (a && b)` is one line but two branch conditions.
  100% line coverage can miss the `a=true, b=false` path entirely.
- **MC/DC (Modified Condition/Decision Coverage)**: required by DO-178C Level A. Each
  condition must independently affect the outcome. For `if (a && b)`:
  - Need: `(T,T)→T`, `(F,T)→F` (a decides), `(T,F)→F` (b decides) — minimum 3 tests.
- **Coverage is necessary but not sufficient**: puzzle01 demonstrates 100% line coverage
  with a real bug. Coverage tells you what you DIDN'T test, not that what you DID test is
  correct.

---

## 7. CI/CD Pipeline for C++ Projects

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI Pipeline                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│  Stage 1 │  Stage 2 │  Stage 3 │  Stage 4 │  Stage 5 │ Stage 6 │
│  Build   │  Test    │  ASan    │  TSan    │ Coverage │  Fuzz   │
│          │          │          │          │          │         │
│ Debug +  │ ctest    │ asan     │ tsan     │ gcov +   │ 5-min   │
│ Release  │ --output │ build +  │ build +  │ lcov     │ libfuzz │
│ compile  │ -on-fail │ full     │ full     │ gate:    │ corpus  │
│          │          │ suite    │ suite    │ ≥80%     │ check   │
│          │          │          │          │          │         │
│ ✓ Both   │ ✓ All   │ ✓ Zero  │ ✓ Zero  │ ✓ ≥80%  │ ✓ Zero  │
│ compile  │ pass     │ errors   │ races    │ lines    │ crashes │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘
         │           │           │           │           │
         ▼           ▼           ▼           ▼           ▼
    ALL GATES MUST PASS ──────────────────────────────► Deploy/
                                                        Package
```

### Build Matrix

A typical GitHub Actions matrix for a C++ project:

```yaml
strategy:
  matrix:
    os: [ubuntu-22.04, ubuntu-24.04]
    compiler: [gcc-12, gcc-13, clang-16, clang-17]
    build_type: [Debug, Release]
```

That's 2×4×2 = 16 combinations. Sanitizer jobs run separately (different compile flags).

### Sanitizer Gates

Each sanitizer is a **hard gate** — any finding blocks the merge:

- **ASan** (Address Sanitizer): heap-buffer-overflow, use-after-free, stack-buffer-overflow,
  memory leaks (`detect_leaks=1`). Runtime overhead: ~2×.
- **TSan** (Thread Sanitizer): data races, lock-order-inversion. Overhead: ~5-15×.
  **Cannot run with ASan simultaneously.**
- **UBSan** (Undefined Behavior Sanitizer): signed overflow, null deref, alignment
  violations, shift out of range. Minimal overhead. Combine with ASan.

### Coverage Reporting

```bash
# In CI, fail if coverage drops below threshold:
lcov ... -o coverage.info
COVERAGE=$(lcov --summary coverage.info 2>&1 | grep 'lines' | grep -oP '\d+\.\d+')
if (( $(echo "$COVERAGE < 80.0" | bc -l) )); then
    echo "Coverage $COVERAGE% < 80% — FAILING"
    exit 1
fi
```

---

## 8. Putting It All Together

The exercises in this week build a complete production setup:

1. **ex01** — A three-tier CMake project with presets, install rules, and cross-compile stub
2. **ex02** — GoogleTest suite for an SPSC queue (concurrent, parameterized, death tests)
3. **ex03** — Fuzzer harness for a binary message parser
4. **ex04** — A profiling target with three intentional performance bugs
5. **ex05** — A full CI-ready project with a `run_all_checks.sh` script

The puzzles demonstrate why individual tools are insufficient:
- **puzzle01**: 100% line coverage hides a real bug (MC/DC gap)
- **puzzle02**: ASan passes but TSan catches a data race (no heap corruption, just UB)

### Recommended Workflow

```
1. Write code
2. Write tests (≥80% coverage)
3. Run ASan build → fix memory bugs
4. Run TSan build → fix races
5. Run UBSan build → fix UB
6. Run fuzzer on parsers/deserializers → fix edge cases
7. Profile → optimize hot paths
8. CI enforces all of the above on every PR
```

This order is deliberate: correctness first (tests + sanitizers), then performance
(profiling). Optimizing buggy code is wasted effort.

---

## References

- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/) — Henry Schreiner
- [GoogleTest Primer](https://google.github.io/googletest/primer.html)
- [libFuzzer Tutorial](https://github.com/google/fuzzing/blob/master/tutorial/libFuzzerTutorial.md)
- Brendan Gregg — [Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [DO-178C and MC/DC](https://en.wikipedia.org/wiki/Modified_condition/decision_coverage)
