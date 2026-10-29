# Control Systems — Learning Plan
### From a bare DC motor to a Jetson↔MCU real-time control stack
### For: Software engineer building and debugging warehouse AMR motor/navigation control

---

## Why This Track Exists

You debug warehouse robots daily. You've seen:
- `cmd_vel` gaps causing the robot to drift mid-tile
- Integral windup after safety monitor releases e-stop → overshoot into a rack
- Motor oscillation when PID gains are pushed too high on a new firmware branch
- The mysterious 10 ms budget between Jetson sending `cmd_vel` and the STM32 executing it

You know the *symptoms*. This track gives you the *theory* to predict them, the *math* to quantify them, and the *code* to reproduce them — on both microcontrollers (fixed-point C) and Jetson (floating-point C++/Python).

---

## Two Worlds, One System

```
┌─────────────────────────────────────────────────────────────────┐
│                      JETSON (Linux, ROS2)                       │
│                                                                 │
│   Nav2 Planner ──→ Controller ──→ cmd_vel @ 20–50 Hz           │
│                                    │                            │
│        Trajectory tracking         │  floating-point            │
│        Pure pursuit / DWB          │  best-effort timing        │
│        Covariance reasoning        │  ~1–10 ms jitter           │
│                                    ▼                            │
│  ─────────────────────── SPI bus (nanopb) ──────────────────── │
│                                    │                            │
│                                    ▼                            │
│                       STM32 (Zephyr RTOS)                       │
│                                                                 │
│   cmd_vel ──→ [Speed PID] ──→ [Current PID] ──→ PWM ──→ Motor │
│                    ↑               ↑                            │
│                 encoder         current sensor                  │
│                 @ 10 kHz        @ 20 kHz                        │
│                                                                 │
│        Fixed-point arithmetic      │  deterministic ISR         │
│        No malloc, no floats        │  <1 µs jitter             │
│        Watchdog: 5 ms timeout      │                            │
└─────────────────────────────────────────────────────────────────┘
```

**This track teaches both layers** and the critical contract between them.

---

## Dependency Graph

```
                    ┌──────────────────────────┐
                    │ 10-advanced-control.md   │
                    │ (feedforward, MPC intro)  │
                    └────────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                      │
┌──────────▼──────────┐ ┌───────▼────────┐ ┌──────────▼──────────┐
│ 08-trajectory-      │ │ 09-two-layer-  │ │ 07-cascaded-        │
│ tracking.md         │ │ architecture   │ │ loops.md            │
│ (pure pursuit, DWB) │ │ (MCU↔Jetson)   │ │ (current→speed→pos) │
└──────────┬──────────┘ └───────┬────────┘ └──────────┬──────────┘
           │                    │                      │
           └────────────────────┼──────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                     │
┌──────────▼──────────┐ ┌──────▼───────┐ ┌──────────▼──────────┐
│ 05-mcu-motor-       │ │ 06-fixed-    │ │ 04-discrete-time-   │
│ control.md          │ │ point-pid.md │ │ control.md          │
│ (PWM, H-bridge,     │ │ (integer C)  │ │ (ZOH, Tustin,       │
│  encoder)           │ │              │ │  sampling)          │
└──────────┬──────────┘ └──────┬───────┘ └──────────┬──────────┘
           │                   │                     │
           └───────────────────┼─────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                    │
┌──────────▼──────────┐ ┌─────▼──────────┐ ┌──────▼──────────────┐
│ 01-what-is-control  │ │ 02-modeling-   │ │ 03-pid-controller   │
│ (feedback concept)  │ │ dc-motors.md   │ │ (PID, tuning,       │
│                     │ │ (plant model)  │ │  anti-windup)       │
└─────────────────────┘ └────────────────┘ └─────────────────────┘
```

**Reading order:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 (strictly sequential)

---

## Topic Breakdown

| # | File | Est. Time | What It Gives You |
|---|------|-----------|-------------------|
| 01 | `01-what-is-control.md` | 3 hrs | Open/closed loop, block diagrams, transfer functions, why feedback works |
| 02 | `02-modeling-dc-motors.md` | 4 hrs | Electrical + mechanical model of a DC motor, step response, back-EMF |
| 03 | `03-pid-controller.md` | 5 hrs | P/PI/PD/PID, Ziegler-Nichols, manual tuning, anti-windup, derivative kick |
| 04 | `04-discrete-time-control.md` | 4 hrs | Continuous→discrete (ZOH, Tustin), Nyquist rate, aliasing, quantization |
| 05 | `05-mcu-motor-control.md` | 4 hrs | PWM, H-bridge circuits, current sensing, quadrature encoder decoding |
| 06 | `06-fixed-point-pid.md` | 4 hrs | Integer-only PID in C, Q16.16 math, timer ISR, no floats, no malloc |
| 07 | `07-cascaded-loops.md` | 4 hrs | Current→speed→position cascade, bandwidth separation, why 3 nested loops |
| 08 | `08-trajectory-tracking.md` | 4 hrs | Pure pursuit, Stanley, DWB, regulated pure pursuit, lookahead tuning |
| 09 | `09-two-layer-architecture.md` | 4 hrs | MCU↔Jetson timing contract, cmd_vel latency budget, watchdog, SPI protocol |
| 10 | `10-advanced-control.md` | 4 hrs | Feedforward, gain scheduling, intro to MPC, model-based vs model-free |

### Debugging Sessions

| # | File | What You Diagnose |
|---|------|-------------------|
| D1 | `debugging/01-motor-oscillation.md` | PID gains too high → limit cycle, diagnose with scope + step response |
| D2 | `debugging/02-integral-windup.md` | Robot hits wall, integrator winds up, e-stop release → overshoot |
| D3 | `debugging/03-cmd-vel-gap.md` | Jetson loop jitter → cmd_vel gap → robot drifts (AMR bag analysis) |

### Exercises (Python + C)

| # | File | What You Build |
|---|------|----------------|
| E1 | `exercises/01-block-diagrams.md` | Reduce block diagrams, compute closed-loop TF by hand |
| E2 | `exercises/02-motor-modeling.md` | Derive motor TF from datasheet, simulate step response |
| E3 | `exercises/03-pid-tuning-sim.md` | Tune PID in Python, plot step/ramp/disturbance response |
| E4 | `exercises/04-discrete-pid-c.md` | Implement discrete PID in C, float vs fixed-point comparison |
| E5 | `exercises/05-cascade-design.md` | Design current→speed cascade, compute bandwidth ratios |
| E6 | `exercises/06-pure-pursuit.md` | Implement pure pursuit in Python, animate on a track |
| E7 | `exercises/07-latency-analysis.md` | Given timing diagram, compute worst-case control delay |
| E8 | `exercises/08-full-pipeline.md` | End-to-end: cmd_vel → SPI → PID → motor → encoder → odom |

### Senior Interview Questions

| # | File | Domain |
|---|------|--------|
| S1 | `senior-questions/01-control-theory-deep.md` | Stability, robustness, gain margins, nonlinear traps |
| S2 | `senior-questions/02-embedded-control.md` | Fixed-point, ISR timing, DMA, RTOS scheduling |
| S3 | `senior-questions/03-system-integration.md` | Two-layer architecture, latency budgets, failure modes |

---

## Prerequisites

| You need | From track | Why |
|----------|------------|-----|
| Op-amp, ADC basics | `electronics/03` | Current sensing, encoder conditioning |
| SPI protocol | `electronics/05` | MCU↔Jetson data link |
| Zephyr timer ISR | `zephyr/01` | Where the PID loop actually runs |
| Odometry math | `navigation-estimator/01` | Feeds the trajectory tracker |
| C++ RTOS patterns | `cpp-advanced/03` | RT scheduling on Jetson side |

---

## Checkpoints (test yourself before moving on)

After each lesson, you should be able to answer its checkpoint questions *from memory*.
If you can't, re-read the relevant section before continuing.

| After lesson | You can... |
|--------------|-----------|
| 01 | Draw the block diagram of a feedback system, explain what "plant", "controller", "error" mean |
| 02 | Write the transfer function of a DC motor from its datasheet parameters |
| 03 | Tune a PID controller using Ziegler-Nichols, explain when integral term helps vs hurts |
| 04 | Convert a continuous PID to discrete-time using bilinear transform, explain why sample rate matters |
| 05 | Explain how PWM drives a motor through an H-bridge, decode a quadrature encoder signal |
| 06 | Implement a PID controller in C using only integer arithmetic, explain Q-format |
| 07 | Design a two-loop cascade (speed over current), explain bandwidth separation rule |
| 08 | Implement pure pursuit, explain lookahead distance vs curvature trade-off |
| 09 | Draw the full MCU↔Jetson timing diagram, identify where 10 ms of latency budget goes |
| 10 | Explain when PID is insufficient and what alternatives exist (feedforward, MPC) |
