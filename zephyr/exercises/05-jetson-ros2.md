# Exercises: Jetson spidev + ROS2 + EKF Integration
### Covers: Projects 9–11 — spidev, RT thread architecture, QoS, timestamps, TF2, robot_localization

---

## Section A — Conceptual Questions

Answer in 2–4 sentences. If you can't answer without looking, re-read the study notes section indicated.

---

**A1.** What path does a `spi.xfer2()` call travel from Python userspace down to physical wire
transitions and back? List every layer in order. Why can't your Python code talk to the SPI
controller hardware directly?

*(Study notes: 1.1)*

---

**A2.** A colleague says: "I just installed JetPack, opened Python, and got `FileNotFoundError` when
trying to `spi.open(0, 0)`. The kernel module must be missing." What is the much more likely cause,
and what is the exact command to fix it?

*(Study notes: 1.2)*

---

**A3.** Explain the `cs_change` field in `spi_ioc_transfer`. Your STM32 implements a two-phase
protocol: one write byte followed by a 4-byte read, with CS staying low for the entire sequence.
Which `cs_change` value do you set on the write transfer struct, and why?

*(Study notes: 1.3)*

---

**A4.** Why does `spi.xfer2()` block your calling thread? What hardware mechanism causes the kernel
to wake your thread back up? Approximately how long does it take to transfer 64 bytes at 10MHz?
Show the calculation.

*(Study notes: 1.5)*

---

**A5.** You have a ROS2 node running on a standard Linux kernel (no PREEMPT-RT). Your timer fires at
10ms intervals. You measure that your `ros2 topic hz /imu/raw` shows 87Hz. Without any other
information, name the two most likely architectural causes (one bad design pattern, one OS
scheduling issue) and the correct fix for each.

*(Study notes: 1.6, 1.10)*

---

**A6.** A classmate argues: "I should use `RELIABLE` QoS for my IMU topic to make sure no messages
are ever dropped." Explain specifically why this is the wrong choice at 100Hz, and describe the
concrete failure mode that will occur when a subscriber falls 100ms behind.

*(Study notes: 1.11)*

---

**A7.** You capture a timestamp with `self.get_clock().now()` at the top of your ROS2 timer
callback, then call `xfer2()`, then decode the protobuf. Is this timestamp correct for the IMU
measurement? Why or why not? What value would be correct, and when exactly should it be captured?

*(Study notes: 1.12)*

---

**A8.** What do the diagonal elements of the `linear_acceleration_covariance` array in a
`sensor_msgs/Imu` message represent? What is the physical unit? What happens to the EKF if you
publish a message with all nine elements set to `0.0`? What does it mean to set element `[0]`
to `-1.0`?

*(Study notes: 1.13)*

---

**A9.** Draw the TF2 frame tree for a simple differential-drive robot with an IMU. Label each edge
with: (a) which node publishes it, (b) whether it is static or dynamic. What breaks if your IMU
message's `frame_id` is `"imu"` but your static transform uses `"imu_link"`?

*(Study notes: 1.14)*

---

**A10.** Explain in your own words what `irqbalance` does and why it is normally a good idea on a
server. Why is it harmful for a real-time SPI acquisition thread? What is the exact failure mode
(describe the sequence of events from irqbalance action to observed timing spike)?

*(Study notes: 1.15)*

---

**A11.** What does `SCHED_FIFO` give you that `SCHED_OTHER` does not? On a system running both a
`SCHED_FIFO priority=90` SPI acquisition thread and normal `SCHED_OTHER` ROS2 executor threads,
describe what happens at the OS level when the SPI acquisition thread's `xfer2()` call returns.

*(Study notes: 1.10)*

---

**A12.** What is `isolcpus`? List the three things you need to do after adding `isolcpus=3` to the
bootloader config to fully isolate your acquisition thread on CPU3. Why is isolating the CPU alone
not sufficient without also pinning the SPI IRQ?

*(Study notes: 1.9)*

---

**A13.** The rosbag2 default storage backend is SQLite. Explain the specific mechanism by which
recording a rosbag degrades your 100Hz sensor publish rate. What storage backend avoids this, and
what command-line flag selects it?

*(Process: think through what SQLite's WAL flush does and when)*

---

**A14.** Why is `CLOCK_MONOTONIC_RAW` preferred over `rclpy`'s `self.get_clock().now()` for
timestamping sensor measurements? Name a concrete scenario where using `get_clock().now()` would
produce a wrong timestamp even if it were called immediately after `xfer2()` returns.

*(Study notes: 1.12)*

---

## Section B — Spot the Bug

Each snippet has one or more bugs. Identify every bug, explain why it is wrong, and write the correct version.

---

**B1.** What is wrong with this spidev setup? (Hint: there are two bugs.)

```python
spi = spidev.SpiDev()
spi.open(0, 0)
spi.mode = 0
# spi.max_speed_hz not set
spi.bits_per_word = 8

def read_frame():
    return bytes(spi.xfer([0x00] * 64))   # xfer, not xfer2
```

---

**B2.** This node publishes IMU data. Identify the architectural bug that causes ~87Hz actual rate
instead of 100Hz. Identify the timestamp bug that causes EKF drift.

```python
class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge')
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 10_000_000
        self.spi.mode = 0
        self.pub = self.create_publisher(Imu, '/imu/raw', 10)
        self.create_timer(0.010, self._cb)

    def _cb(self):
        raw = self.spi.xfer2([0x00] * 64)           # <--- line A
        stamp = self.get_clock().now().to_msg()       # <--- line B
        frame = SensorFrame()
        frame.ParseFromString(bytes(raw))
        msg = Imu()
        msg.header.stamp = stamp
        self.pub.publish(msg)
```

For each labelled line, say: what is wrong, what is the effect, and how to fix it.

---

**B3.** This covariance setting will silently crash the EKF. What is wrong?

```python
msg = Imu()
msg.linear_acceleration.x = frame.imu.accel_x
msg.linear_acceleration.y = frame.imu.accel_y
msg.linear_acceleration.z = frame.imu.accel_z
msg.angular_velocity.x = frame.imu.gyro_x
# Covariance — "don't know, leave default"
# msg.linear_acceleration_covariance is not set → all zeros by default
self.imu_pub.publish(msg)
```

---

**B4.** The TF broadcaster below has a bug that will cause `robot_localization` to silently ignore
the IMU. Find it.

```python
tf = TransformStamped()
tf.header.stamp    = self.get_clock().now().to_msg()
tf.header.frame_id = 'base_link'
tf.child_frame_id  = 'imu_link'
tf.transform.translation.x = 0.05
tf.transform.rotation.w    = 1.0
broadcaster.sendTransform(tf)

# ... in the SPI publisher node:
msg.header.frame_id = 'imu'   # <--- bug
```

---

**B5.** This `robot_localization` YAML snippet has two errors that will prevent the EKF from using
the IMU. Find them.

```yaml
imu0: /imu_raw              # <--- A

imu0_config:
  - [false, false, false,
     true,  true,  true,
     true,  true,  true,
     false, false, false,
     false, false, false]

process_noise_covariance:
  - [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    # <--- B: [0,0] = 0.0
     # ... (rest is fine)
```

---

**B6.** Spot the race condition between the acquisition thread and the publisher thread:

```python
# Global list (mutable)
latest_frame = None

def acquisition_thread():
    while True:
        raw = spi.xfer2([0]*64)
        global latest_frame
        latest_frame = raw           # writes global

def timer_callback(self):
    if latest_frame is not None:
        frame = SensorFrame()
        frame.ParseFromString(bytes(latest_frame))  # reads global
        # ... publish
```

---

**B7.** After setting up SCHED_FIFO, a developer runs:

```bash
python3 spi_bridge.py &
ros2 topic hz /imu/raw
# Shows: average rate: 87.435Hz
```

They check their RT setup:

```bash
sudo chrt -p $(pgrep -f spi_bridge)
# pid 12345's scheduling policy: SCHED_FIFO
# pid 12345's scheduling priority: 90
```

RT priority looks correct. What other RT setup step is missing that explains the 87Hz rate?

---

**B8.** This command is supposed to run the SPI bridge at real-time priority pinned to CPU3:

```bash
chrt -f 90 taskset -c 3 python3 spi_bridge.py
```

The `spi_bridge.py` spawns a new thread for acquisition. Will that thread run at SCHED_FIFO on CPU3?
Why or why not? How must the code be changed?

---

## Section C — Fill in the Blank

Fill in the missing values or code. Every blank is a single expression, number, or short code fragment.

---

**C1.** A 64-byte SPI frame at 10MHz takes _______ microseconds to transfer (show arithmetic:
64 × 8 bits ÷ 10,000,000 bits/sec × 1,000,000).

---

**C2.** Complete this spidev setup:

```python
spi = spidev.SpiDev()
spi.open(_______, _______)    # bus=0, cs=0
spi.max_speed_hz = _______    # 10 MHz
spi.mode = _______            # CPOL=0 CPHA=0
```

---

**C3.** Fill in the QoS profile for a sensor topic that should never back-pressure the publisher:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
sensor_qos = QoSProfile(
    reliability = ReliabilityPolicy._______ ,   # don't retry dropped packets
    durability  = DurabilityPolicy._______ ,   # don't store for late joiners
    history     = HistoryPolicy.KEEP_LAST,
    depth       = _______,                     # keep only the 5 most recent
)
```

---

**C4.** Fill in the covariance fields to tell `robot_localization` to use its own YAML config
for IMU covariance (not the message values):

```python
msg.linear_acceleration_covariance[0] = _______   # indicates "unknown"
msg.angular_velocity_covariance[0]    = _______   # same
msg.orientation_covariance[0]         = _______   # IMU has no absolute orientation
```

---

**C5.** Fill in the SCHED_FIFO setup:

```python
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

SCHED_______  = 1   # scheduling policy constant

class SchedParam(ctypes.Structure):
    _fields_ = [('sched_priority', ctypes.c_int)]

param = SchedParam(sched_priority=_______)   # use priority 90
libc.sched_setscheduler(_______, SCHED_FIFO, ctypes.byref(param))
# first arg: 0 means "_______"
```

---

**C6.** Complete the `cyclictest` command that runs 100,000 iterations across 4 threads at maximum
RT priority, with memory locked, showing a summary:

```bash
sudo cyclictest _______ -sp_______ -t_______ -l_______
```

And the `strace` command to measure wall-clock time of every ioctl in `spi_reader.py`:

```bash
sudo strace _______ -e trace=_______ python3 spi_reader.py 2>&1 | grep _______
```

---

## Section D — Lab Tasks

Complete each task on your actual hardware (Jetson + STM32). Include the verification output as
specified. If you don't have hardware, simulate where possible and note what would differ on real
hardware.

---

**D1. Baseline hardware validation before writing any ROS2 code**

Before writing a single line of Python bridge code, validate the SPI link using the kernel's
built-in test tool.

Steps:
1. Run `jetson-io.py`, configure SPI1, reboot
2. Verify `/dev/spidev0.0` exists
3. Run `sudo spidev_test -D /dev/spidev0.0 -s 10000000 -p "AABBCCDD" -v`

**Verification criteria:**
- `ls /dev/spidev0.0` succeeds (file exists)
- `spidev_test` exits with no error
- If STM32 is in loopback or echoes bytes: RX matches TX
- Record: what did the RX bytes show? If you see all 0xFF, what does that indicate?

**Write-up:** Take a screenshot or paste the `spidev_test` output. Note the SPI clock speed it
negotiated and whether the result is consistent across 5 runs.

---

**D2. Measure your RT latency baseline and compare standard vs PREEMPT-RT**

If you have both a standard and PREEMPT-RT kernel available (or can switch):

Steps:
1. Boot standard kernel: `sudo cyclictest -m -sp99 -t4 -l10000`
2. Boot PREEMPT-RT kernel: same command
3. Under PREEMPT-RT: start your ROS2 SPI node, then run cyclictest again

**Verification criteria:**
```
Standard kernel:  Max latency should be in range 500µs – 3ms
PREEMPT-RT:       Max latency should be < 100µs (Jetson Orin)
With ROS2 node:   Max should not increase by more than 50µs vs idle
```

Record the exact Max values for each scenario in a table. If Max exceeds 500µs on PREEMPT-RT,
investigate: check `sudo systemctl status irqbalance` and disable if running.

---

**D3. Reproduce and fix the "ioctl in timer callback" rate degradation**

Steps:
1. Write (or study) the WRONG version from study notes section 2.4 (xfer2 inside timer callback)
2. Run it and measure: `ros2 topic hz /imu/raw`
3. Note the measured Hz (should be ~83–90Hz)
4. Switch to the CORRECT version (thread + queue)
5. Measure rate again: `ros2 topic hz /imu/raw`

**Verification criteria after fix:**
```bash
ros2 topic hz /imu/raw
# Expected output:
# average rate: 99.938
#         min: 0.009s max: 0.011s std dev: 0.00014s window: 1000
```

The rate must be within ±0.5Hz of 100Hz. Standard deviation must be <0.0005s (0.5ms jitter).

**Write-up:** Paste the `ros2 topic hz` output for both versions. Did the rate improvement match
your expectation? What else changed (CPU usage, latency)?

---

**D4. Diagnose and fix a QoS mismatch**

Steps:
1. Start your SPI bridge node with `qos_profile_sensor_data` (BEST_EFFORT)
2. Write a subscriber with RELIABLE QoS:
   ```python
   from rclpy.qos import QoSProfile, ReliabilityPolicy
   reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
   sub = self.create_subscription(Imu, '/imu/raw', self.callback, reliable_qos)
   ```
3. Verify the subscriber receives nothing
4. Use `ros2 topic info -v /imu/raw` to diagnose the mismatch
5. Fix the subscriber QoS to match

**Verification criteria:**
- Step 3: `ros2 topic hz /imu/raw` at the publisher shows 100Hz, but no output from the subscriber
- Step 4: `ros2 topic info -v` shows incompatible Reliability policies between pub and sub
- Step 5: After fix, subscriber callback fires at 100Hz; no data loss

**Write-up:** Paste the output of `ros2 topic info -v /imu/raw` showing both the incompatible and
compatible configurations.

---

**D5. End-to-end EKF validation**

Steps:
1. Start your SPI bridge node (with correct architecture and timestamps)
2. Start the static TF broadcaster for `base_link → imu_link`
3. Start `robot_localization` with an `ekf_config.yaml` (see study notes 2.7)
4. Run the verification commands below

**Verification criteria:**
```bash
# Check 1: IMU rate
ros2 topic hz /imu/raw
# Expected: 99.5–100.5 Hz

# Check 2: End-to-end latency
ros2 topic delay /imu/raw
# Expected: < 5ms

# Check 3: EKF output exists
ros2 topic hz /odometry/filtered
# Expected: 100Hz (EKF republishes at same rate as inputs)

# Check 4: TF tree is complete
ros2 run tf2_tools view_frames && evince frames.pdf
# Expected: map → odom → base_link → imu_link all connected

# Check 5: No NaN in EKF output
ros2 topic echo /odometry/filtered --once | grep -i nan
# Expected: no output (no NaN values)

# Check 6: rosbag doesn't degrade rate
ros2 bag record --storage mcap /imu/raw &
ros2 topic hz /imu/raw
# Expected: still 99.5–100.5 Hz (no degradation from recording)
```

**Write-up:** Paste all six command outputs. If the EKF produces NaN: check your covariance fields.
If rate degrades under recording: check your storage backend and QoS.

---

## Section E — System Design

These are open-ended questions. There is no single correct answer — quality of reasoning matters more
than the specific design chosen.

---

**E1. Draw the complete thread model**

Draw a block diagram showing:
- Every thread in the system (acquisition thread, ROS2 executor, any others)
- For each thread: scheduling policy, priority, CPU affinity
- The data path between threads (what data structure, max size, what happens when full)
- At what point in the data path is the timestamp captured

Include on your diagram: which blocks run at SCHED_FIFO vs SCHED_OTHER, and why.

If the timestamp is captured in the wrong thread, annotate what error this introduces.

---

**E2. Design a timing budget for 100Hz operation**

You need the full pipeline (SPI transfer → decode → publish → EKF integration) to complete in <10ms.

Allocate a time budget for each stage:

| Stage | Your budget (µs) | Why |
|-------|-----------------|-----|
| SPI transfer (64 bytes @ 10MHz) | | |
| Kernel scheduling latency (PREEMPT-RT) | | |
| Protobuf decode | | |
| Message fill + publish | | |
| DDS transport to EKF | | |
| EKF integration step | | |
| **Total** | **< 10000µs** | |

Where does the timestamp need to be captured to be within your budget? What happens to the EKF if
the "protobuf decode" step is moved before the timestamp capture?

---

**E3. Design a watchdog for the SPI link**

Consider this failure mode: the STM32 firmware crashes or reboots. While it's restarting, its SPI
output is invalid (CS may be left asserted, or all bytes are 0xFF). Your ROS2 bridge keeps publishing
`/imu/raw` with zero or garbage values. The EKF happily fuses the garbage and the robot's estimated
position diverges.

Design a watchdog that detects and responds to STM32 failure. Your design should address:

1. **Detection:** How do you distinguish "STM32 is responding but quiet" from "STM32 firmware crash"?
   (Hint: what field in the protobuf schema could help? What threshold?)

2. **Handling:** What does the bridge node do when it detects a STM32 failure?
   - Option A: stop publishing IMU messages
   - Option B: publish with a "validity" flag
   - Option C: publish with inflated covariance
   Which is best for `robot_localization`, and why? (Hint: what does -1.0 in covariance[0] mean?)

3. **Recovery:** The STM32 restarts 2 seconds later and starts sending valid frames again.
   How does the bridge detect "valid again"? What should it do?

4. **Implementation:** Write a pseudocode sketch of the watchdog logic (no need for full
   working code — describe the state machine and the transitions).

---

**E4. QoS policy selection for a mixed-criticality system**

Your robot has these ROS2 topics:

| Topic | Rate | Criticality | Consumers |
|-------|------|-------------|-----------|
| `/imu/raw` | 100Hz | Safety-critical | EKF, log |
| `/cmd_vel` | 10Hz | Safety-critical (stale = crash) | Motor driver |
| `/map` | 1Hz | High (localization depends on it) | AMCL |
| `/debug_image` | 30Hz | Low (just for operators) | rviz2 |
| `/battery_status` | 1Hz | Medium | Dashboard |
| `/emergency_stop` | Event | CRITICAL (must never be lost) | All safety nodes |

For each topic, choose: `RELIABLE` or `BEST_EFFORT`, `VOLATILE` or `TRANSIENT_LOCAL`, and a queue
`depth`. Justify each choice in one sentence. Consider: what is the failure mode if a message is
dropped? What is the failure mode if old messages are replayed to a new subscriber?

---

**E5. Plan the robot_localization EKF sensor fusion troubleshooting workflow**

Your EKF (`/odometry/filtered`) diverges — the estimated position drifts in a straight line even
when the robot is stationary.

Design a systematic debugging workflow (step-by-step) that a junior engineer could follow to
diagnose whether the cause is:

a) Wrong timestamp on IMU messages
b) Zero covariance (all zeros, not -1.0)
c) frame_id mismatch (TF lookup failure)
d) irqbalance causing 2ms timestamp spikes
e) QoS mismatch causing the EKF to never receive IMU data

For each cause, specify:
- The exact `ros2` command that will reveal it
- What "sick" output looks like vs "healthy" output
- The fix (one line or one command)

Your workflow should not require touching any source code until you've narrowed it to one root cause.

---

## Answer Hints (for self-checking)

<details>
<summary>A1 hint</summary>
The path is: Python → spidev library → ioctl syscall → Linux kernel SPI core → Tegra SPI controller driver → DMA engine → physical MOSI/CLK/CS pins → MISO receives → DMA interrupt → kernel wakes process → ioctl returns → Python. Direct access to hardware is prevented by CPU privilege levels (ring 0 vs ring 3).
</details>

<details>
<summary>A4 hint — calculation</summary>
64 bytes × 8 bits/byte = 512 bits. 512 / 10,000,000 = 0.0000512 seconds = 51.2 µs. Add kernel scheduling overhead: ~55–60µs total.
</details>

<details>
<summary>B1 hints</summary>
Bug 1: `max_speed_hz` not set → defaults to 500kHz kernel default → 20× too slow, can't sustain 100Hz with 64-byte frames. Bug 2: `spi.xfer()` (not `xfer2`) pulses CS between each byte → STM32 sees 64 separate transactions instead of one.
</details>

<details>
<summary>B2 hints</summary>
Line A: `xfer2()` is a blocking ioctl call inside the executor's timer callback. The executor is frozen for ~55µs while SPI transfers. All other callbacks queue. At 100Hz this produces ~87Hz actual rate. Fix: move to dedicated acquisition thread. Line B: timestamp captured after `xfer2()` returns AND after the executor scheduling delay — may be 0.5–3ms late. For angular velocity at 100Hz, 2ms error = 0.002 rad phantom orientation change per sample. Fix: capture `CLOCK_MONOTONIC_RAW` in the acquisition thread immediately after `xfer2()` completes.
</details>

<details>
<summary>B3 hint</summary>
When covariance is all zeros, `R = 0` in the Kalman gain equation `K = PH^T(HPH^T + R)^-1`. Division by zero → NaN propagates through the entire EKF state vector. Fix: set `msg.linear_acceleration_covariance[0] = -1.0` (unknown) or fill the 3×3 diagonal with realistic variance values like 0.01.
</details>

<details>
<summary>B6 hint</summary>
`latest_frame` is a Python global list written by one thread and read by another without a lock. Short lists in CPython are "atomic" to reassign (the GIL protects the reference swap), but `bytes(latest_frame)` reads the list contents across multiple operations — if the writer is simultaneously replacing it, you can read a partially-updated list. The correct fix is to use `queue.Queue(maxsize=1)` or a `threading.Lock`. The `queue` approach is better because it also prevents the "stale frame" problem.
</details>

<details>
<summary>C1 answer</summary>
64 bytes × 8 bits/byte = 512 bits. 512 bits ÷ 10,000,000 bits/sec = 0.0000512 seconds = **51.2 microseconds**.
</details>

<details>
<summary>D3 expected Hz after fix</summary>
After moving xfer2() to a dedicated thread: `ros2 topic hz /imu/raw` should show 99.8–100.2 Hz. The timer callback is now <5µs, so no executor stall occurs.
</details>
