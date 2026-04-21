# Week 8 — Capstone: Mini Flight Software

## Overview

This capstone project integrates every major concept from Weeks 1–7 into a
self-contained **mini flight software** system: a cyclic executive running
sensor acquisition, PID control, telemetry, health monitoring, and a software
watchdog — all wired through a mode manager state machine.

The system simulates a single-axis attitude controller reading an IMU, running
a PID loop, logging telemetry over a lock-free queue, and responding to faults
by transitioning through degraded modes — exactly the kind of architecture you
find in CubeSat, drone, and AMR firmware.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.cpp                            │
│  CLI parsing · RT setup · component wiring · shutdown   │
└──────────┬──────────────────────────────────────────────┘
           │ owns
           ▼
┌──────────────────┐     events      ┌──────────────────┐
│   ModeManager    │◄────────────────│  HealthMonitor   │
│ Boot→Init→Nom→   │                 │ rate checks,     │
│ Degraded→SafeStop│                 │ missed heartbeat │
└────────┬─────────┘                 └───────▲──────────┘
         │ guards task execution              │ heartbeat()
         ▼                                    │
┌──────────────────────────────────────────────────────────┐
│                  CyclicExecutive (1 ms frame)            │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌──────┐ │
│  │SensorSim │  │PIDControl  │  │ Telemetry │  │Health│ │
│  │ 1 kHz    │  │  500 Hz    │  │  100 Hz   │  │10 Hz │ │
│  └────┬─────┘  └─────┬──────┘  └─────┬─────┘  └──────┘ │
│       │read()        │update()       │                   │
│       └──────────────┘               │                   │
│              pid_output              │                   │
│                  │         ┌─────────┘                   │
│                  ▼         ▼                             │
│            ┌──────────────────┐                          │
│            │   SPSCQueue      │──► Logger thread (cout)  │
│            └──────────────────┘                          │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐                    │
│  │  Watchdog    │   │ TraceBuffer  │                    │
│  │  (separate   │   │ (ring, RT-   │                    │
│  │   thread)    │   │  safe write) │                    │
│  └──────────────┘   └──────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

---

## Component List

| Component          | Header                          | Week Origin | Purpose                                      |
|--------------------|---------------------------------|-------------|----------------------------------------------|
| `ModeManager`      | `mode_manager.hpp`              | W3, W7      | `std::variant` state machine, Hamming IDs     |
| `CyclicExecutive`  | `cyclic_executive.hpp`          | W6          | RT loop, jitter tracking, task shedding       |
| `SensorSim`        | `sensor_sim.hpp`                | W5          | Simulated IMU with fault injection            |
| `PIDController`    | `pid_controller.hpp`            | W7          | CRTP-based PID with anti-windup               |
| `SPSCQueue`        | `spsc_queue.hpp`                | W2          | Lock-free single-producer single-consumer     |
| `Watchdog`         | `watchdog.hpp`                  | W4, W6      | Software watchdog, separate thread            |
| `HealthMonitor`    | `health_monitor.hpp`            | W5          | Heartbeat rate monitoring, degradation trigger |
| `TraceBuffer`      | `trace_buffer.hpp`              | W7          | Fixed-size ring buffer, RT-safe, post-mortem  |

---

## Build Instructions

### Quick build (single compilation unit)

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread \
    -Iinclude src/main.cpp -o flight_sw
```

### CMake build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Sanitizer presets

```bash
# Address Sanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build-asan -j$(nproc)

# Thread Sanitizer
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread"
cmake --build build-tsan -j$(nproc)

# UB Sanitizer
cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=all"
cmake --build build-ubsan -j$(nproc)
```

---

## How to Run

### Normal mode (5 seconds, nominal)

```bash
./flight_sw --duration 5
```

### Fault injection (sensor stuck after 2s)

```bash
./flight_sw --duration 10 --fault-after 2
```

### Stress test (tight loop, maximum jitter measurement)

```bash
./flight_sw --duration 30 --stress
```

### Full demo (fault + stress)

```bash
./flight_sw --duration 20 --fault-after 5 --stress
```

The program prints a final report including:
- Jitter histogram (μs buckets)
- Mode transition log
- Health events (missed heartbeats, fault detections)
- Trace buffer dump (last N events)
- Task execution counts

---

## Quality Checklist

Before considering this capstone "done", verify:

- [ ] **JPL Rule compliance**: no heap after init, bounded loops, no recursion
      in the cyclic executive path. `new`/`malloc` only during setup.
- [ ] **ASan clean**: zero leaks, zero out-of-bounds under normal + fault modes.
- [ ] **TSan clean**: zero data races. The SPSC queue, trace buffer, and watchdog
      all use atomics correctly with appropriate memory orderings.
- [ ] **UBSan clean**: no signed overflow, no null dereference, no misaligned
      access (especially in the SPSC queue).
- [ ] **Jitter target**: 95th-percentile frame jitter < 100 μs on a desktop
      Linux box (non-RT kernel). Under SCHED_FIFO on an RT kernel, target < 20 μs.
- [ ] **Fault recovery**: injecting STUCK sensor triggers Degraded mode within
      the health monitor's detection window (< 1s at default settings).
- [ ] **Watchdog fires**: if the cyclic executive stalls (e.g., infinite loop
      in a task), the watchdog callback fires within its timeout.
- [ ] **Trace completeness**: the trace buffer captures the full mode transition
      sequence and at least the last 1024 events.

---

## Concepts Exercised

| Week | Topic                          | Where Used                              |
|------|--------------------------------|-----------------------------------------|
| W1   | Value semantics, moves         | Config structs, SensorReading            |
| W2   | Lock-free, atomics             | SPSCQueue, TraceBuffer                  |
| W3   | std::variant state machine     | ModeManager                             |
| W4   | Threads, synchronization       | Watchdog, logger thread                 |
| W5   | Error handling, fault modes    | SensorSim fault injection, HealthMonitor|
| W6   | RT patterns, jitter            | CyclicExecutive, mlockall, SCHED_FIFO   |
| W7   | CRTP, safety patterns          | PIDController, Hamming state IDs         |

---

## Extensions (if you want more)

1. **Add a Kalman filter** between sensor and PID using Eigen (header-only).
2. **Binary telemetry** — write SPSC output to a binary file, parse offline.
3. **Multiple axes** — extend to 3-axis attitude control (roll/pitch/yaw).
4. **Hardware-in-the-loop** — replace SensorSim with a real serial IMU reader.
5. **Formal verification** — model the ModeManager in TLA+ or UPPAAL.
