# Day 47: Contributing to Apache TVM

> **Phase III · Week 7 · Day 47 of 70 · 2.5 hours**

*"The fastest way to understand a compiler is to break it, fix it, and submit the patch."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 46: MLC-LLM](day-46-mlc-llm.md) | [Day 48: Compiler Testing & Verification](day-48-compiler-testing-verification.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You've spent weeks **using** TVM — importing models, writing schedules, tuning kernels. Now it's time to shift from consumer to contributor. Apache TVM is an open-source project with 900+ contributors, governed by the Apache Software Foundation. Contributing isn't just altruistic — it's the best way to deepen your understanding. When you fix a bug in TIR lowering or add a schedule primitive, you must truly understand the invariants the codebase enforces. This lesson covers the complete contribution workflow: building from source, navigating the codebase, writing tests, and submitting a pull request that actually gets merged.

---

## 1. TVM Community & Governance

### The Apache Way

TVM graduated as an **Apache Top-Level Project** in 2022. Apache governance has specific rules:

```
Apache TVM Governance Structure
═══════════════════════════════

  PMC (Project Management Committee)
  ├── Votes on releases, new committers
  ├── Legal/licensing oversight
  └── ~20 members (Tianqi Chen, Junru Shao, etc.)

  Committers
  ├── Merge rights to main branch
  ├── Earned through sustained contribution
  └── ~80 committers

  Contributors
  ├── Anyone who submits a PR
  ├── No special permissions required
  └── 900+ total contributors

  Decision Process:
  ┌─────────────────────────────────────────────────┐
  │  Bug fix / small change  →  1 committer review  │
  │  New feature / API change → RFC + 2+ reviews    │
  │  Architecture change      → Discuss thread + vote│
  │  Release                  → PMC vote (majority)  │
  └─────────────────────────────────────────────────┘

  Communication Channels:
  • discuss.tvm.apache.org  — design discussions, RFCs
  • GitHub Issues            — bug reports, feature requests
  • GitHub PRs               — code review
  • dev@tvm.apache.org       — official mailing list (releases)
```

### The RFC Process

For non-trivial changes, TVM uses **Request for Comments (RFCs)**:

```
RFC Lifecycle
═════════════

  1. Post RFC on discuss.tvm.apache.org
     └─ Template: Motivation → Proposed Design → Alternatives → Timeline

  2. Community feedback (≥ 72 hours)
     └─ Committers and PMC members comment

  3. Iterate on design based on feedback
     └─ May require multiple rounds

  4. Merge RFC as a tracking document
     └─ Implementation follows in separate PRs

  Examples of things that need RFCs:
  ✓ New IR (Relax replaced Relay via RFC)
  ✓ New pass infrastructure
  ✓ Major API changes
  ✓ New backend target

  Things that do NOT need RFCs:
  ✗ Bug fixes
  ✗ Documentation improvements
  ✗ Test additions
  ✗ Performance improvements to existing passes
```

---

## 2. Building TVM from Source

### Prerequisites and Build

```bash
# 1. Clone the repository (with submodules!)
git clone --recursive https://github.com/apache/tvm.git
cd tvm

# 2. Install dependencies (Ubuntu)
sudo apt-get install -y python3-dev python3-pip \
    llvm-17-dev libllvm17 cmake ninja-build \
    libtinfo-dev zlib1g-dev libedit-dev

# 3. Create build directory and configure
mkdir build && cd build

# 4. Copy the config template
cp ../cmake/config.cmake .

# 5. Edit config.cmake — enable what you need
#    Key options:
#    set(USE_LLVM  "/usr/bin/llvm-config-17")   ← CPU codegen
#    set(USE_CUDA  ON)                           ← NVIDIA GPU
#    set(USE_METAL OFF)                          ← macOS GPU
#    set(USE_VULKAN ON)                          ← Vulkan GPU
#    set(USE_RELAY_DEBUG ON)                     ← Debug mode

# 6. Build with ninja (faster than make)
cmake -G Ninja .. && ninja -j$(nproc)

# 7. Install Python package in development mode
cd ../python
pip install -e . --user
```

### Build Configuration Matrix

```
Build Options Cheat Sheet
═════════════════════════

  Option              │ When to enable           │ Build time impact
  ────────────────────┼──────────────────────────┼──────────────────
  USE_LLVM            │ Always (CPU codegen)     │ +30s
  USE_CUDA            │ NVIDIA GPU dev           │ +60s
  USE_RELAY_DEBUG     │ Debugging passes         │ +5s (runtime cost)
  USE_MICRO           │ µTVM / embedded          │ +20s
  USE_PROFILER        │ Performance analysis     │ +10s
  USE_VULKAN          │ Vulkan GPU target        │ +40s
  HIDE_PRIVATE_SYMBOLS│ Release builds           │ neutral
  ────────────────────┼──────────────────────────┼──────────────────
  Total (all on)      │ ~15–25 min from scratch  │
  Incremental rebuild │ Typically 10–60 seconds  │
```

### Verifying Your Build

```python
# Quick smoke test after build
import tvm
print(f"TVM version: {tvm.__version__}")
print(f"LLVM enabled: {tvm.runtime.enabled('llvm')}")
print(f"CUDA enabled: {tvm.runtime.enabled('cuda')}")

# Verify Relay works
from tvm import relay
x = relay.var("x", shape=(1, 3, 224, 224))
y = relay.nn.conv2d(x, relay.var("w"), kernel_size=(3, 3), padding=(1, 1),
                    channels=64)
mod = tvm.IRModule.from_expr(y)
print(f"Relay module created: {mod}")
```

---

## 3. Code Organization Tour

### Repository Layout

```
tvm/
├── src/                     ← C++ core (the "real" compiler)
│   ├── ir/                  ← Base IR nodes (Expr, Stmt, Type)
│   ├── tir/                 ← TIR: low-level loop IR
│   │   ├── transforms/      ← TIR passes (vectorize, unroll, etc.)
│   │   └── schedule/        ← Schedule primitives
│   ├── relay/               ← Relay: graph-level IR
│   │   ├── op/              ← Operator definitions (conv2d, etc.)
│   │   ├── transforms/      ← Relay passes (FuseOps, FoldConstant)
│   │   └── backend/         ← Graph/AOT executors
│   ├── relax/               ← Relax: next-gen IR
│   │   ├── op/              ← Relax operators
│   │   └── transform/       ← Relax passes
│   ├── target/              ← Target descriptions (cuda, llvm, etc.)
│   ├── runtime/             ← Runtime: NDArray, Module, RPC
│   └── auto_scheduler/      ← Ansor auto-scheduling
│
├── python/tvm/              ← Python bindings (mirrors src/)
│   ├── ir/                  ← Python IR wrappers
│   ├── tir/                 ← TIR Python API
│   ├── relay/               ← Relay Python API
│   ├── relax/               ← Relax Python API
│   ├── meta_schedule/       ← MetaSchedule Python API
│   └── contrib/             ← Third-party integrations
│
├── tests/python/            ← Python tests (primary test suite)
│   ├── relay/               ← Relay tests
│   ├── tir/                 ← TIR tests
│   ├── relax/               ← Relax tests
│   └── contrib/             ← Integration tests
│
├── include/tvm/             ← C++ headers (public API)
├── 3rdparty/                ← Vendored dependencies (dlpack, dmlc-core)
├── apps/                    ← Example applications
├── docs/                    ← Sphinx documentation
└── cmake/                   ← Build system configuration
```

### The C++ ↔ Python Bridge (PackedFunc)

Every TVM API call crosses the C++/Python boundary through `PackedFunc`:

```
How Python Calls C++
════════════════════

  Python side                    C++ side
  ─────────────                  ────────────
  tvm.relay.transform            TVM_REGISTER_GLOBAL(
    .FuseOps()                     "relay._transform.FuseOps")
        │                              │
        └──── PackedFunc ──────────────┘
              • Type-erased function pointer
              • Arguments: TVMArgs (variant type)
              • Return: TVMRetValue
              • Registered via TVM_REGISTER_GLOBAL macro

  Key insight: Most "Python" code is just thin wrappers
  calling C++ through PackedFunc. To change behavior,
  you usually modify C++ code, not Python.
```

---

## 4. Writing Tests

### TVM Testing Conventions

TVM uses **pytest** with custom utilities in `tvm.testing`:

```python
import tvm
import tvm.testing
from tvm import relay, tir
import numpy as np

# Convention: one test file per feature/pass
# File: tests/python/relay/test_pass_fold_constant.py

def test_fold_constant_simple():
    """Test that constant expressions are folded at compile time."""
    # Arrange: build a Relay graph with constant operands
    c1 = relay.const(np.array([1.0, 2.0, 3.0], dtype="float32"))
    c2 = relay.const(np.array([4.0, 5.0, 6.0], dtype="float32"))
    expr = relay.add(c1, c2)

    # Act: run the FoldConstant pass
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.FoldConstant()(mod)

    # Assert: result should be a single constant
    result = mod["main"].body
    assert isinstance(result, relay.Constant)
    np.testing.assert_allclose(
        result.data.numpy(),
        np.array([5.0, 7.0, 9.0], dtype="float32"),
    )


# Parametric testing across targets
@tvm.testing.parametrize_targets("llvm", "cuda")
def test_conv2d_correctness(target, dev):
    """Test conv2d produces correct output on each backend."""
    data = relay.var("data", shape=(1, 3, 8, 8), dtype="float32")
    weight = relay.var("weight", shape=(16, 3, 3, 3), dtype="float32")
    out = relay.nn.conv2d(data, weight, padding=(1, 1))

    mod = tvm.IRModule.from_expr(out)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    runtime_mod = tvm.contrib.graph_executor.GraphModule(
        lib["default"](dev)
    )

    data_np = np.random.uniform(size=(1, 3, 8, 8)).astype("float32")
    weight_np = np.random.uniform(size=(16, 3, 3, 3)).astype("float32")
    runtime_mod.set_input("data", data_np)
    runtime_mod.set_input("weight", weight_np)
    runtime_mod.run()

    tvm_out = runtime_mod.get_output(0).numpy()
    # Compare against numpy reference
    # (simplified; real test uses tvm.testing.assert_allclose)
    assert tvm_out.shape == (1, 16, 8, 8)
```

### Running Tests

```bash
# Run specific test file
pytest tests/python/relay/test_pass_fold_constant.py -v

# Run a single test function
pytest tests/python/relay/test_pass_fold_constant.py::test_fold_constant_simple -v

# Run only CPU tests (skip CUDA)
pytest tests/python/relay/ -m "not requires_cuda" -v

# Run with TIR debug checks enabled
TVM_LOG_DEBUG="tir=1" pytest tests/python/tir/ -v

# Lint check (required before PR)
python3 -m pylint python/tvm --rcfile=pylintrc
bash tests/lint/check_file_type.sh
```

---

## 5. Submitting a Pull Request

### Development Workflow

```
PR Workflow — From Fork to Merge
════════════════════════════════

  1. Fork apache/tvm on GitHub

  2. Create a feature branch
     $ git checkout -b fix-relay-fold-constant

  3. Make changes (code + tests)

  4. Run local checks:
     $ pytest tests/python/relay/test_pass_fold_constant.py -v
     $ python3 -m pylint python/tvm/relay/transform.py
     $ clang-format -i src/relay/transforms/fold_constant.cc

  5. Write commit message:
     ┌──────────────────────────────────────────────┐
     │ [Relay] Fix FoldConstant for dynamic shapes  │
     │                                              │
     │ Previously, FoldConstant would crash when    │
     │ encountering relay.Any() in tensor shapes.   │
     │ This patch adds a shape check before folding.│
     │                                              │
     │ Fixes #12345                                 │
     └──────────────────────────────────────────────┘

  6. Push and open PR
     $ git push origin fix-relay-fold-constant

  7. CI runs (~1–2 hours):
     ├── Lint (clang-format, pylint, mypy)
     ├── Build (Linux CPU, Linux CUDA, macOS, Windows)
     ├── Unit tests (CPU + GPU)
     └── Integration tests

  8. Address reviewer feedback
     └── Respond to every comment (resolve or discuss)

  9. Squash-merge after approval (1+ committer)
```

### Commit Message Convention

```
Format:  [Component] Short description

Components:
  [Relay]         — Relay IR, passes, ops
  [TIR]           — TIR IR, schedule, transforms
  [Relax]         — Relax IR
  [Runtime]       — Runtime, NDArray, RPC
  [Target]        — Target descriptions, codegen
  [MetaSchedule]  — Auto-tuning
  [CI]            — Build system, CI config
  [Docs]          — Documentation
  [Test]          — Test infrastructure

Examples:
  [TIR] Add vectorization support for ARM SVE
  [Relay][Quantization] Fix scale folding for depthwise conv2d
  [Runtime] Reduce memory allocation in graph executor
```

---

## 6. Common Contribution Areas for Beginners

### Where to Start

```
Beginner-Friendly Contribution Areas
═════════════════════════════════════

  Difficulty │ Area                    │ What to do
  ──────────┼─────────────────────────┼──────────────────────────
  ★☆☆☆☆     │ Documentation           │ Fix typos, add examples
  ★☆☆☆☆     │ Test coverage           │ Add tests for uncovered ops
  ★★☆☆☆     │ Error messages          │ Improve unclear error msgs
  ★★☆☆☆     │ Python type hints       │ Add type annotations
  ★★★☆☆     │ Operator implementation │ Add a missing relay op
  ★★★☆☆     │ Bug fixes               │ Fix issues labeled "good
            │                         │  first issue"
  ★★★★☆     │ Pass improvements       │ Optimize existing passes
  ★★★★☆     │ New schedule primitive  │ Add to TIR scheduler
  ★★★★★     │ New backend             │ BYOC integration
  ★★★★★     │ IR design               │ Relax improvements
```

### Example: Adding a Test for an Uncovered Operator

```python
# File: tests/python/relay/test_op_nn.py (append)

def test_layer_norm_fp16():
    """Verify LayerNorm produces correct output in float16.

    Regression test for issue #XXXXX where fp16 LayerNorm
    produced NaN on certain input distributions.
    """
    data = relay.var("data", shape=(2, 4, 8), dtype="float16")
    gamma = relay.var("gamma", shape=(8,), dtype="float16")
    beta = relay.var("beta", shape=(8,), dtype="float16")

    out = relay.nn.layer_norm(data, gamma, beta, axis=-1)
    mod = tvm.IRModule.from_expr(out)

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    dev = tvm.cpu(0)
    rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Use values that historically triggered NaN
    data_np = np.random.uniform(-10, 10, (2, 4, 8)).astype("float16")
    gamma_np = np.ones(8, dtype="float16")
    beta_np = np.zeros(8, dtype="float16")

    rt.set_input("data", data_np)
    rt.set_input("gamma", gamma_np)
    rt.set_input("beta", beta_np)
    rt.run()

    result = rt.get_output(0).numpy()
    # Must not contain NaN or Inf
    assert np.all(np.isfinite(result)), f"Non-finite values: {result}"
    # Check output has zero mean and unit variance (within fp16 tolerance)
    tvm.testing.assert_allclose(
        result.mean(axis=-1),
        beta_np[0],
        atol=0.1,  # fp16 tolerance
    )
```

---

## Hands-On Exercises

### Exercise 1: Build TVM from Source (30 min)

```bash
# Clone, configure (CPU-only for speed), build, and verify
git clone --recursive https://github.com/apache/tvm.git
cd tvm && mkdir build && cd build
cp ../cmake/config.cmake .
# Edit: set(USE_LLVM "/usr/bin/llvm-config-17")
cmake -G Ninja .. && ninja -j$(nproc)
cd ../python && pip install -e .

# Verify:
python3 -c "import tvm; print(tvm.__version__)"
```

### Exercise 2: Navigate the Codebase (20 min)

Answer these by reading the source:

1. Where is `relay.nn.conv2d` defined in Python? In C++?
2. What file contains the `FuseOps` pass implementation?
3. How does `tvm.testing.parametrize_targets` work?
4. Find the `PackedFunc` registration for `relay.build`.

### Exercise 3: Write and Run a Test (30 min)

Write a test for `relay.nn.batch_norm` that:
- Tests with `float32` input of shape `(4, 16, 8, 8)`
- Verifies output shape matches input shape
- Checks output is finite (no NaN/Inf)
- Runs on `"llvm"` target

### Exercise 4: Simulate a PR (20 min)

1. Find an open issue labeled `good first issue` on GitHub
2. Read the issue and linked code
3. Draft a commit message following TVM conventions
4. Identify which test file(s) you would modify

---

## Key Takeaways

1. **TVM follows Apache governance** — RFCs for big changes, committer review for all PRs, PMC votes for releases
2. **Building from source** is required for development; enable only the targets you need to keep build times under 10 minutes
3. **The codebase mirrors C++ ↔ Python** — `src/relay/` maps to `python/tvm/relay/`, connected through `PackedFunc`
4. **Tests are mandatory** — every PR must include tests; use `tvm.testing.parametrize_targets` for multi-backend coverage
5. **Start with documentation, tests, or error messages** — these are the fastest path to your first merged PR
6. **CI takes 1–2 hours** — run local tests first (`pytest -v`) to avoid wasting CI cycles on obvious failures

---

## Further Reading

- [TVM Contributor Guide](https://tvm.apache.org/docs/contribute/) — official onboarding
- [TVM Design & Architecture](https://tvm.apache.org/docs/arch/) — architectural docs
- [Good First Issues](https://github.com/apache/tvm/labels/good%20first%20issue) — beginner tasks
- [TVM Discussion Forum](https://discuss.tvm.apache.org/) — RFC discussions, community Q&A
- [Apache Way Documentation](https://www.apache.org/theapacheway/) — governance philosophy
- [TVM CI Dashboard](https://ci.tlcpack.ai/) — build status

---

## Tomorrow: Compiler Testing & Verification

Day 48 examines a critical question: **how do you know your compiler produces correct code?** You'll learn about numerical accuracy testing, fuzzing strategies, differential testing across backends, and why `np.allclose` with the wrong tolerance has caused more silent bugs than any code review can catch.
