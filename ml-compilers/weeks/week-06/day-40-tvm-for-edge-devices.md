# Day 40: TVM for Edge Devices

> **Phase III · Week 6 · Day 40 of 70 · 2.5 hours**

*"The real test of a compiler isn't how fast it runs on the machine that built it — it's how fast it runs on the machine in your pocket, your car, or your thermostat."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 39: Quantization in TVM](day-39-quantization-in-tvm.md) | [Day 41: TVM Unity & Relax](day-41-tvm-unity-relax.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Most ML models are trained on beefy GPU workstations, but they *run* on edge devices: phones, drones, robots, IoT sensors, Raspberry Pis, NVIDIA Jetsons, and bare-metal microcontrollers. These targets have wildly different ISAs, memory budgets (kilobytes to gigabytes), and OS capabilities (Linux, Zephyr RTOS, or no OS at all). TVM's **cross-compilation** workflow lets you compile on your x86 development machine and produce binaries for ARM Cortex-A, Cortex-M, RISC-V, or Hexagon DSPs. The **µTVM** (MicroTVM) subsystem goes further — deploying to devices with no OS and as little as 256 KB of flash. Mastering edge deployment is the difference between a model that lives in a notebook and one that ships in a product.

---

## 1. Cross-Compilation Fundamentals

### The Host/Target Split

In cross-compilation, **two machines** are involved:

| Role | Description | Example |
|:--|:--|:--|
| **Host** | Where compilation happens | x86-64 workstation, Ubuntu 22.04 |
| **Target** | Where the compiled model runs | ARM Cortex-A72 (Raspberry Pi 4) |

TVM cleanly separates these through its `Target` string and the RPC (Remote Procedure Call) system:

```
┌─────────────────────────────┐       ┌──────────────────────────┐
│        HOST (x86-64)        │       │     TARGET (ARM)         │
│                             │       │                          │
│  ┌───────────┐              │  RPC  │  ┌──────────────────┐   │
│  │ TVM       │   compile    │ ────▶ │  │ tvm_rpc_server   │   │
│  │ Compiler  │──────────▶   │       │  │                  │   │
│  └───────────┘  .tar module │       │  │  load & execute  │   │
│                             │       │  └──────────────────┘   │
│  ┌───────────┐              │       │                          │
│  │ AutoTune  │   tune via   │ ◀───▶ │  measure real latency   │
│  │ Engine    │   RPC        │       │                          │
│  └───────────┘              │       └──────────────────────────┘
└─────────────────────────────┘
```

### Target Specification Strings

The TVM target string encodes the ISA, features, and runtime:

```python
# Raspberry Pi 4 (Cortex-A72, ARMv8-A, NEON)
target_rpi4 = tvm.target.Target(
    "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu "
    "-mattr=+neon,+fp-armv8,+crc"
)

# NVIDIA Jetson Nano (Maxwell GPU + ARM CPU)
target_jetson_gpu = tvm.target.Target("cuda -arch=sm_53")
target_jetson_cpu = tvm.target.Target(
    "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
)

# Cortex-M4 microcontroller (no OS, Thumb-2 ISA)
target_m4 = tvm.target.Target(
    "c -keys=arm_cpu -mcpu=cortex-m4 -march=armv7e-m "
    "-mfloat-abi=hard -mfpu=fpv4-sp-d16 -runtime=c -system-lib"
)

# Hexagon DSP (Qualcomm Snapdragon)
target_hexagon = tvm.target.Target("hexagon -mcpu=v68")
```

---

## 2. The Cross-Compilation Workflow

### Step-by-Step: x86 → Raspberry Pi 4

```python
import tvm
from tvm import relay
import onnx

# 1. Load model on HOST
onnx_model = onnx.load("mobilenet_v2.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1, 3, 224, 224)})

# 2. Define TARGET
target = tvm.target.Target(
    "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
)

# 3. Compile on HOST for TARGET
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 4. Export as a deployable artifact
lib.export_library(
    "mobilenet_v2_rpi4.tar",
    cc="/usr/bin/aarch64-linux-gnu-gcc"  # Cross-compiler toolchain
)
```

### Deploying on the Target

```python
# On the Raspberry Pi 4
import tvm
from tvm.contrib import graph_executor
import numpy as np

# Load the cross-compiled module
lib = tvm.runtime.load_module("mobilenet_v2_rpi4.tar")
dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))

# Run inference
input_data = np.random.uniform(size=(1, 3, 224, 224)).astype("float32")
module.set_input("input", input_data)
module.run()
output = module.get_output(0).numpy()
```

---

## 3. Graph Executor vs AOT Executor

TVM provides two execution strategies with fundamentally different tradeoffs:

### Graph Executor (Interpreter-Style)

```
┌──────────────────────────────────────────────────┐
│              Graph Executor Runtime               │
│                                                   │
│  ┌──────────┐    ┌──────────────────────┐        │
│  │  JSON    │    │   Shared Library     │        │
│  │  Graph   │──▶ │   (.so / .tar)       │        │
│  │  Desc.   │    │   - kernel functions │        │
│  └──────────┘    │   - fused operators  │        │
│       │          └──────────────────────┘        │
│       ▼                     │                    │
│  Interpreter reads          │                    │
│  graph node by node ───▶ calls kernels           │
│                                                   │
│  Needs: dynamic memory allocator, JSON parser     │
│  Size:  ~300 KB runtime overhead                  │
└──────────────────────────────────────────────────┘
```

### AOT Executor (Ahead-of-Time Compiled)

```
┌──────────────────────────────────────────────────┐
│              AOT Executor                         │
│                                                   │
│  ┌──────────────────────────────────────┐        │
│  │  Single C function: tvmgen_default_run()      │
│  │                                      │        │
│  │  // All calls are static, no graph   │        │
│  │  tvmgen_default_fused_conv2d(args);  │        │
│  │  tvmgen_default_fused_relu(args);    │        │
│  │  tvmgen_default_fused_dense(args);   │        │
│  └──────────────────────────────────────┘        │
│                                                   │
│  No interpreter, no JSON, no dynamic alloc        │
│  Size:  ~10 KB overhead (+ kernel code)           │
└──────────────────────────────────────────────────┘
```

### Comparison Table

| Feature | Graph Executor | AOT Executor |
|:--|:--|:--|
| Runtime overhead | ~300 KB | ~10 KB |
| Dynamic allocation | Required | Optional (static planning) |
| Graph description | JSON at runtime | Baked into C code |
| Debugging | Inspect graph nodes | Step through C |
| OS requirement | Linux/RTOS with malloc | Bare-metal compatible |
| Best for | Raspberry Pi, Jetson | Cortex-M, Arduino, Zephyr |

### Enabling AOT Compilation

```python
# AOT compilation with static memory planning
with tvm.transform.PassContext(opt_level=3):
    executor = relay.backend.Executor("aot", {
        "unpacked-api": True,      # Simple C calling convention
        "interface-api": "packed",  # or "c" for minimal runtime
    })
    runtime = relay.backend.Runtime("crt")  # C Runtime (no libc needed)
    
    lib = relay.build(
        mod, target=target_m4, params=params,
        executor=executor, runtime=runtime
    )
```

---

## 4. µTVM — TVM for Microcontrollers

### The Microcontroller Challenge

Microcontrollers (MCUs) operate under extreme constraints:

| Resource | Typical MCU | Raspberry Pi 4 | Ratio |
|:--|:--|:--|:--|
| RAM | 256 KB – 1 MB | 4 GB | 4,000–16,000× |
| Flash | 512 KB – 2 MB | 32 GB (SD) | 16,000–65,000× |
| Clock | 80–400 MHz | 1.5 GHz (4 cores) | 4–19× |
| OS | None / Zephyr / FreeRTOS | Linux | — |

µTVM addresses these with:
- **AOT executor** (no interpreter overhead)
- **Static memory planning** (all buffers pre-allocated at compile time)
- **C-runtime (CRT)** — a minimal runtime (~10 KB) with no OS dependency
- **Model Library Format (MLF)** — a self-contained archive for integration

### µTVM Architecture

```
┌───────────────────────────── HOST ────────────────────────────┐
│                                                                │
│  Python Script ──▶ Relay IR ──▶ TIR ──▶ AOT C Code            │
│                                           │                    │
│  ┌──────────────────────────────────────┐ │                    │
│  │ Project API                          │ │                    │
│  │  - generate_project()                │◀┘                    │
│  │  - build()                           │                      │
│  │  - flash()                           │                      │
│  │  - open_transport() ←── serial/JTAG  │                      │
│  └──────────────────────────────────────┘                      │
│           │              ▲                                     │
└───────────│──────────────│─────────────────────────────────────┘
            │ flash        │ serial I/O
            ▼              │
┌──────────────────────────────────────────────────┐
│  MICROCONTROLLER (e.g. STM32F746, nRF5340)       │
│                                                   │
│  ┌────────────────────────────────────┐           │
│  │  Generated AOT code               │           │
│  │  + CRT runtime (~10 KB)           │           │
│  │  + Zephyr/Arduino platform glue   │           │
│  │  + Input/output buffers           │           │
│  └────────────────────────────────────┘           │
│                                                   │
│  Flash: 512 KB    RAM: 256 KB    Clock: 216 MHz   │
└──────────────────────────────────────────────────┘
```

### µTVM with Arduino

```python
import tvm
from tvm import relay
from tvm.micro import export_model_library_format

# Load a tiny model (e.g., keyword spotting, 50 KB)
mod, params = relay.frontend.from_tflite(tflite_model)

target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m4")

with tvm.transform.PassContext(opt_level=3):
    executor = relay.backend.Executor("aot", {"unpacked-api": True})
    runtime = relay.backend.Runtime("crt")
    lib = relay.build(mod, target=target, params=params,
                      executor=executor, runtime=runtime)

# Export as Model Library Format
export_model_library_format(lib, "keyword_spotting.tar")

# Generate Arduino project
from tvm.micro.project_api.client import ProjectAPIClient
project = tvm.micro.generate_project(
    template_dir="arduino",              # or "zephyr"
    module=lib,
    generated_project_dir="./arduino_project",
    options={"board": "nano33ble", "project_type": "example_project"}
)
project.build()
project.flash()
```

---

## 5. Memory Planning for Constrained Devices

### The Problem

A model with 20 intermediate tensors doesn't need 20 buffers simultaneously — some tensors' lifetimes don't overlap:

```
Tensor lifetimes (time →):

  T0: ████░░░░░░░░░░░░░░░░
  T1: ░░████░░░░░░░░░░░░░░
  T2: ░░░░████░░░░░░░░░░░░
  T3: ░░░░████████░░░░░░░░
  T4: ░░░░░░░░░░████░░░░░░
  T5: ░░░░░░░░░░░░████░░░░
  T6: ░░░░░░░░░░░░░░████░░

  Without planning: 7 buffers = 7 × 16 KB = 112 KB
  With planning:    3 buffers = 3 × 16 KB =  48 KB  (57% saving)
  
  Buffer reuse:
    Buffer A: T0, T2, T4, T6  (non-overlapping)
    Buffer B: T1, T5           (non-overlapping)
    Buffer C: T3               (overlaps with T2, T4)
```

### Static Memory Planning in TVM

```python
# Enable the USMP (Unified Static Memory Planner)
with tvm.transform.PassContext(opt_level=3, config={
    "tir.usmp.enable": True,
    "tir.usmp.algorithm": "hill_climb",  # or "greedy_by_size"
}):
    lib = relay.build(mod, target=target, params=params,
                      executor=relay.backend.Executor("aot"),
                      runtime=relay.backend.Runtime("crt"))
```

The **USMP** solves a memory coloring problem:

$$\text{minimize} \quad \sum_{b \in \text{buffers}} \text{size}(b)$$
$$\text{subject to} \quad \forall (T_i, T_j) \text{ with overlapping lifetimes: } \text{buf}(T_i) \neq \text{buf}(T_j)$$

Available algorithms:
- **Greedy by size** — assign largest tensors first, $O(n^2)$
- **Hill climb** — iterative improvement over greedy, better packing
- **Linear scan** — similar to register allocation in compilers

---

## 6. RPC Testing on Real Hardware

### Setting Up the RPC Tracker

The TVM RPC system lets your host machine communicate with remote devices:

```bash
# On the HOST: start the RPC tracker
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# On the TARGET (Raspberry Pi / Jetson):
python -m tvm.exec.rpc_server \
    --tracker=HOST_IP:9190 \
    --key=rpi4 \
    --port=9090
```

### Remote Compilation + Benchmarking

```python
import tvm
from tvm import relay, autotvm
from tvm.contrib import utils, ndk
from tvm import rpc

# Connect to RPC tracker
tracker = rpc.connect_tracker("HOST_IP", 9190)
remote = tracker.request("rpi4", priority=0, session_timeout=60)

# Upload and run on remote device
temp = utils.tempdir()
path = temp.relpath("mobilenet.tar")
lib.export_library(path, cc="/usr/bin/aarch64-linux-gnu-gcc")

remote.upload(path)
rlib = remote.load_module("mobilenet.tar")
dev = remote.cpu(0)

# Create remote executor
from tvm.contrib import graph_executor
module = graph_executor.GraphModule(rlib["default"](dev))

# Benchmark
import numpy as np
input_data = np.random.uniform(size=(1, 3, 224, 224)).astype("float32")
module.set_input("input", input_data)

ftimer = module.module.time_evaluator("run", dev, repeat=10, number=10)
prof_res = np.array(ftimer().results) * 1000  # Convert to ms
print(f"Mean: {np.mean(prof_res):.2f} ms, Std: {np.std(prof_res):.2f} ms")
```

### Remote AutoTuning

```python
# Tune on real hardware via RPC
from tvm import meta_schedule as ms

database = ms.tune_tvm(
    mod=mod,
    target=target_rpi4,
    config=ms.TuneConfig(
        strategy="evolutionary",
        num_trials_per_iter=32,
        max_trials_per_task=200,
    ),
    runner=ms.runner.RPCRunner(
        rpc_config=ms.runner.RPCConfig(
            tracker_host="HOST_IP",
            tracker_port=9190,
            tracker_key="rpi4",
            session_timeout_sec=60,
        ),
        max_workers=1,  # One device
    ),
    work_dir="./tune_rpi4",
)
```

---

## Hands-On Exercises

### Exercise 1: Cross-Compile for ARM (20 min)

Cross-compile a small model without a physical device using QEMU:

```bash
# Install ARM cross-compiler and QEMU
sudo apt install gcc-aarch64-linux-gnu qemu-user-static

# Verify
aarch64-linux-gnu-gcc --version
qemu-aarch64-static --version
```

```python
import tvm
from tvm import relay
import numpy as np

# Build a tiny model
x = relay.var("x", shape=(1, 64), dtype="float32")
w = relay.var("w", shape=(10, 64), dtype="float32")
y = relay.nn.dense(x, w)
y = relay.nn.softmax(y)
mod = tvm.IRModule.from_expr(y)

target = tvm.target.Target(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target)

lib.export_library(
    "tiny_model_arm.so",
    cc="/usr/bin/aarch64-linux-gnu-gcc"
)
print("✓ Cross-compiled for ARM64")
# Test with: qemu-aarch64-static -L /usr/aarch64-linux-gnu ./run_model
```

### Exercise 2: AOT vs Graph Executor Comparison (25 min)

Compare the two executors for a keyword-spotting model:

```python
import tvm
from tvm import relay

# Create a keyword-spotting-sized model
data = relay.var("data", shape=(1, 1, 49, 10), dtype="float32")  # MFCC features
conv = relay.nn.conv2d(data, relay.var("w1", shape=(16, 1, 3, 3)),
                       padding=(1, 1), channels=16, kernel_size=(3, 3))
pool = relay.nn.global_avg_pool2d(conv)
flat = relay.nn.batch_flatten(pool)
dense = relay.nn.dense(flat, relay.var("w2", shape=(12, 16)))  # 12 keywords
out = relay.nn.softmax(dense)

mod = tvm.IRModule.from_expr(out)
target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m4")

# Graph executor build
with tvm.transform.PassContext(opt_level=3):
    lib_graph = relay.build(mod, target=target)

# AOT executor build
with tvm.transform.PassContext(opt_level=3):
    executor = relay.backend.Executor("aot", {"unpacked-api": True})
    runtime = relay.backend.Runtime("crt")
    lib_aot = relay.build(mod, target=target,
                          executor=executor, runtime=runtime)

# Compare: inspect generated code sizes, memory requirements
```

### Exercise 3: Memory Planning Analysis (15 min)

Analyze memory savings from USMP:

```python
# Enable USMP with verbose logging
with tvm.transform.PassContext(opt_level=3, config={
    "tir.usmp.enable": True,
    "tir.usmp.algorithm": "hill_climb",
    "tir.disable_vectorize": True,
}):
    lib = relay.build(mod, target=target,
                      executor=relay.backend.Executor("aot"),
                      runtime=relay.backend.Runtime("crt"))

# Extract memory statistics
# Look for workspace size in the generated code
print("Memory pools allocated:")
# Check: tvmgen_default_workspace_pools in generated artifacts
```

---

## Key Takeaways

1. **Cross-compilation** separates host (compile) from target (run) — TVM's `Target` string specifies ISA, features, and ABI
2. **AOT executor** eliminates graph interpretation overhead (~10 KB vs ~300 KB), enabling bare-metal deployment
3. **µTVM** targets microcontrollers via AOT + CRT runtime + Project API for Arduino/Zephyr integration
4. **Memory planning (USMP)** solves a coloring problem to reuse buffers across non-overlapping tensor lifetimes — critical when RAM is measured in kilobytes
5. **RPC testing** lets you tune and benchmark on real target hardware from the comfort of your host machine
6. **Target strings** are your contract with the backend — wrong `-mattr` flags mean missing NEON optimizations or illegal instructions

---

## Further Reading

- [µTVM: TVM on Microcontrollers](https://tvm.apache.org/docs/topic/microtvm/) — official µTVM documentation
- [TVM RPC System](https://tvm.apache.org/docs/how_to/deploy/rpc.html) — remote deployment guide
- [AOT Compilation in TVM](https://tvm.apache.org/docs/arch/aot_compilation.html) — architecture deep dive
- [Model Library Format](https://tvm.apache.org/docs/arch/model_library_format.html) — packaging for microcontrollers
- [Deploying TVM on Raspberry Pi](https://tvm.apache.org/docs/how_to/deploy/arm_compute_lib.html) — step-by-step ARM deployment
- [USMP: Unified Static Memory Planner (RFC)](https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planner/) — memory planning design

---

## Tomorrow: TVM Unity & Relax

Day 41 looks at the **future of TVM** — the Unity initiative that unifies Relay, TE, and TIR into a single composable framework, and **Relax**, the next-generation IR with first-class dynamic shapes and dataflow blocks. You'll see how the lessons from Days 29–40 feed directly into TVM's next chapter.
