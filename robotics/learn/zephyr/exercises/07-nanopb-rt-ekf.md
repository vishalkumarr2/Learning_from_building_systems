# Exercises: ZBus+nanopb Bridge, Jetson RT Setup, and EKF Integration
### Covers: Projects 7–9 — nanopb framing, Jetson RT tuning, EKF covariance, TF2
**These exercises assume you have read `09-zbus-nanopb-bridge.md`, `10-jetson-rt-setup.md`, and `11-ekf-integration.md`.**

---

## Section A — ZBus + nanopb Framing Questions

---

**A1.** Your `.proto` file defines a message with one `float velocity = 1;` field. You serialize
a `velocity = 0.0f` message with nanopb. The encoded output is **0 bytes**. Explain:
- Why does proto3 produce zero bytes for `0.0f`?
- Why does this break your length-prefix framing protocol?
- Give the exact C code change that prevents this (without changing the protocol buffer format).

---

**A2.** You use a 4-byte little-endian length prefix before each serialized protobuf message.
The receiver reads the 4-byte prefix first, allocates a buffer, then reads exactly that many
bytes. Your test sends a 23-byte payload and a 47-byte payload back-to-back over SPI.

Describe exactly what the receiver's state machine should look like. What are the states?
What data is stored between state transitions? What happens if the FIFO drains mid-payload
(bytes 1–20 arrive, then 10ms silence, then bytes 21–23 arrive)?

---

**A3.** nanopb uses fixed-size arrays. Your `.proto` defines `repeated float readings = 2;`
and you set the nanopb option to `max_count: 12`. Your proto message in C has this struct:

```c
typedef struct {
    pb_size_t readings_count;
    float readings[12];
} SensorData;
```

You receive a message where the Jetson sent 15 readings. What happens in nanopb during
`pb_decode()`? Does it crash, truncate, return an error, or produce undefined behavior?
How would you detect this case?

---

**A4.** Explain the difference between:
- `ZBUS_CHAN_DEFINE` with a `.validator = NULL` 
- `ZBUS_CHAN_DEFINE` with a validator function

When would you add a validator? Give a concrete example for an IMU channel (what invalid
data would it reject?).

---

**A5.** You have three ZBus channels: `imu_chan` (100Hz), `odom_chan` (50Hz), `gps_chan` (1Hz).
Your packer thread subscribes to `imu_chan`. On each new IMU message, it checks the other
two channels for their latest values using a non-blocking peek.

Write the correct Zephyr API calls to:
1. Subscribe packer thread to `imu_chan` using a message queue (not semaphore)
2. Read latest value from `odom_chan` without waiting
3. Safely copy the current GPS value even though it might be stale

---

## Section B — Jetson RT Kernel Questions

---

**B1.** You configure `isolcpus=3` in the kernel boot parameters. Explain:
- What does this prevent?
- What does this NOT prevent?
- How do you actually assign your RT thread to CPU 3?
- Write the exact `taskset` command to pin a running process by PID to CPU 3

---

**B2.** You run `cyclictest -l 10000 -p 99 -i 1000 -a 3` on your Jetson Orin NX. Interpret this
output:

```
T: 0 ( 1234) P:99 I:1000 C:10000 Min:   4 Act:   8 Avg:  11 Max:  427
```

- What does each field mean?
- Is a 427µs maximum latency acceptable for a 100Hz SPI loop (10ms period)?
- What would you investigate to reduce the maximum latency below 100µs?

---

**B3.** Explain the difference between:
- `SCHED_FIFO` with priority 99
- `SCHED_RR` with priority 99
- Default `SCHED_OTHER`

Which should you use for your SPI polling thread, and why? What happens if two `SCHED_FIFO`
priority-99 threads both try to run on CPU 3 simultaneously?

---

**B4.** You run `sudo nvpmodel -m 0 && sudo jetson_clocks` before benchmarking. Why? What does
`MAXN` mode change compared to mode 1 or 2? What thermal risk does this introduce, and what
should you add to your test procedure?

---

**B5.** `irqbalance` is running by default on your Jetson. Explain what it does and why you
disable it before RT testing. After disabling it, what must you do to ensure your RT CPU is
not handling any hardware IRQs?

---

**B6.** You measure ioctl latency using `clock_gettime(CLOCK_MONOTONIC_RAW, ...)` before and
after `ioctl(fd, SPI_IOC_MESSAGE(1), &xfer)`. You see:

```
min: 52µs  mean: 61µs  max: 3847µs  p99: 98µs
```

The max is 63× the mean. Is this acceptable? What is causing the outlier? Write the
`cyclictest` command you would run in parallel to confirm the hypothesis.

---

## Section C — EKF Integration Questions

---

**C1.** Your `robot_localization` EKF node produces this `odom` output:

```
pose:
  covariance: [0, 0, 0, 0, 0, 0, ...]  # all zeros
```

What does a zero covariance mean mathematically? What will happen to subsequent sensor fusion
steps that try to use this covariance? What field in your `ekf.yaml` causes this?

---

**C2.** The `imu0_config` boolean vector in `ekf.yaml` has 15 elements. List what each group
of elements controls:
- Elements 0–2:
- Elements 3–5:
- Elements 6–8:
- Elements 9–11:
- Elements 12–14:

For a 6-DOF IMU (accelerometer + gyroscope, no magnetometer), which elements should be `true`?
Write the complete `imu0_config` line.

> **Hint for AMR context:** Enabling accelerometer fusion on a flat-floor AMR is a known
> footgun. Explain why a Z-axis accelerometer reading from a robot crossing a tile gap can
> trigger `COLLISION_DETECTED`, and whether this means accel fusion should be enabled or
> disabled in `imu0_config` for a ground robot.

---

**C3.** Your EKF outputs NaN in `pose.position.x` after 30 seconds of operation. List three
causes of EKF divergence that produce NaN specifically (as opposed to just poor accuracy).
For each, describe the diagnostic check.

---

**C4.** The TF2 tree for your robot is: `map → odom → base_link → imu_link`.

For each frame pair, answer:
1. Which node publishes this transform?
2. Is it static or dynamic?
3. What happens to the EKF if this link is missing?

---

**C5.** Explain the difference between setting `differential: true` vs `false` in `robot_localization`
for your wheel odometry. You have a differential drive robot with no absolute position source.
Should `odom0_config` include position XY as `true` or `false`? Why?

---

**C6.** You are tuning the `imu0_noise` covariance in `ekf.yaml`. You increase the linear
acceleration noise value from `0.01` to `1.0`. Describe what changes in the EKF behavior:
- Does the filter trust the IMU more or less?
- Does the filter output become smoother or more reactive?
- In which direction does this change the "lag" in the position estimate?

---

## Section D — Cross-Project Integration

---

**D1.** Your SPI bridge runs at 100Hz. The EKF runs at 30Hz. The GPS runs at 1Hz. Draw the
data flow from raw sensor to `robot_localization` output. At each step, identify:
- The transport mechanism (ZBus channel, ROS2 topic, etc.)
- The frequency
- What happens to the data in between (decimation, buffering, none?)

---

**D2.** Your Jetson detects a 200ms dropout in SPI frames (STM32 crashed and rebooted).
During the dropout:
- What does `robot_localization` output?
- What does rviz2 show?
- What should your system do to handle graceful recovery?

---

**D3.** You have `use_control: true` in `ekf.yaml` and your system publishes `cmd_vel`. Explain:
- What does this add to the state vector?
- What is the risk if `cmd_vel` has significant latency (e.g., from WiFi control)?
- In which application scenario would enabling this improve accuracy?

---

## Section E — Debugging Drills

---

**E1.** Your EKF node starts but `/odom` topic has no messages. Write the exact sequence of
ROS2 commands to diagnose this without reading source code.

---

**E2.** The EKF runs but the robot drifts 0.5m after a 360° rotation that should return to the
origin. This is a known EKF limitation. Explain:
- What causes accumulated yaw error?
- What sensor would eliminate this error?
- At what frequency must it be fused?

---

**E3.** You set `rviz2`'s `Fixed Frame` to `map` but the robot marker is not visible.
You change it to `odom` and the robot appears. Explain what this tells you about your
TF2 tree and how to fix it.

---

## Answers

<details>
<summary>A1</summary>

In proto3, default values (0, 0.0, false, empty string) are never encoded on the wire —
this is the "implicit field presence" rule that differs from proto2. A `velocity = 0.0f`
message encodes to a zero-length buffer. With length-prefix framing, a 4-byte prefix of
`0x00000000` followed by zero payload bytes is valid, but many receivers interpret a 0-byte
payload as "end of stream" or a bug. Fix: use `google.protobuf.FloatValue` wrapper (which
always encodes), or add a sequence number / message ID field that is never 0, or switch to
proto2 `optional float` with `has_field` tracking.

</details>

<details>
<summary>A5</summary>

```c
// 1. Subscribe with message queue
ZBUS_MSG_SUBSCRIBER_DEFINE(packer_sub, 4);
zbus_chan_add_obs(&imu_chan, &packer_sub, K_FOREVER);

// 2. Non-blocking peek of latest odom value
struct OdometryMsg odom;
zbus_chan_read(&odom_chan, &odom, K_NO_WAIT);  // returns -EAGAIN if no data

// 3. Read latest GPS value (may be stale - that's OK)
struct GpsMsg gps;
zbus_chan_read(&gps_chan, &gps, K_NO_WAIT);
```

`zbus_chan_read` with `K_NO_WAIT` returns the last published value without waiting. It
returns `-EAGAIN` if the channel has never been published to, which you should handle.

</details>

<details>
<summary>B2</summary>

- `T:0` — thread 0
- `(1234)` — PID
- `P:99` — SCHED_FIFO priority
- `I:1000` — interval 1000µs (1ms)
- `C:10000` — 10,000 iterations completed
- `Min:4` — best latency: 4µs
- `Act:8` — most recent: 8µs
- `Avg:11` — average: 11µs
- `Max:427` — worst-case: 427µs

For a 10ms SPI loop, 427µs is ~4% of the period — technically fine but tight.
For a 1ms IMU read, it would be unacceptable.
Investigate: system IRQs not redirected away from CPU 3, frequency scaling enabled,
`irqbalance` still running, GPU driver interrupts.

</details>

<details>
<summary>C1</summary>

Zero covariance means infinite certainty — the filter claims to know the position with
zero uncertainty. This is mathematically incorrect and will cause the Kalman gain for
any subsequent sensor update to collapse toward zero (the filter stops accepting new
measurements). The culprit in `ekf.yaml` is likely the initial `process_noise_covariance`
or `initial_estimate_covariance` having zero diagonal elements, or publishing an `Odometry`
message with a zero-filled covariance matrix from your driver code.

</details>

<details>
<summary>C2</summary>

The 15-element `imu0_config` vector controls:
- [0–2]: x, y, z position (rarely used for IMU)
- [3–5]: roll, pitch, yaw (orientation)
- [6–8]: vx, vy, vz (linear velocity)
- [9–11]: vroll, vpitch, vyaw (angular velocity — from gyro)
- [12–14]: ax, ay, az (linear acceleration — from accelerometer)

For a 6-DOF IMU: `[false, false, false, true, true, true, false, false, false, true, true, true, true, true, false]`
(orientation from fused estimation, angular velocity from gyro, linear accel from accelerometer,
not yaw accel which would require magnetometer).

</details>
