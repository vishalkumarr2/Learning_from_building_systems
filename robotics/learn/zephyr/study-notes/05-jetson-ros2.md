# Jetson spidev + ROS2 + EKF Integration — Study Notes
### Projects 9–11: spidev, ROS2 Publisher Node, EKF Pipeline
**Hardware:** Jetson Orin NX · JetPack 6.x · PREEMPT-RT kernel · STM32 SPI Slave (from Projects 7–8)

---

## PART 1 — ELI15 Concept Explanations

---

### 1.1 What spidev is — the Linux kernel SPI userspace interface

**The analogy: a postal sort facility with a window hatch**

The SPI controller inside the Jetson's Tegra SoC is a piece of hardware that only the kernel can talk to
directly — like a postal sort facility deep in a government building. You, as a userspace program, are not
allowed in the back room.

`spidev` is the **window hatch** the kernel provides so your Python or C program can hand a message over
and get one back, without you ever needing to know how the sorting machinery works.

Concretely:
- The kernel's SPI controller driver registers a character device at `/dev/spidev<bus>.<cs>`
- `bus` = which SPI controller (0, 1, 2…) — the Jetson has multiple
- `cs`  = which chip-select line (0, 1…) — one physical line per slave device

When you open `/dev/spidev0.0` in Python, you are opening a file descriptor to this hatch. When you call
`xfer2()`, you are:
1. Writing your TX bytes and a configuration struct into the kernel via `ioctl`
2. The kernel driver programs the hardware SPI controller's DMA registers
3. The SPI controller physically toggles CLK, shifts out MOSI, and samples MISO
4. The kernel copies the received MISO bytes back into your buffer
5. `ioctl` returns — you now have the reply

**Important mental model:** from your Python code's perspective, `spidev` looks like a simple file. Under
the hood, the path is:
```
Python xfer2()
  → ioctl(fd, SPI_IOC_MESSAGE, &transfer_struct)
    → Linux kernel SPI core
      → Tegra SPI controller driver
        → Hardware DMA engine shifts bits over MOSI/MISO/CLK
      ← DMA complete interrupt fires
    ← Kernel unblocks your process
  ← ioctl() returns
← xfer2() returns received bytes
```

**The device path tells you the hardware:**
```bash
ls /dev/spidev*
# /dev/spidev0.0  →  SPI controller 0, chip select 0  (our STM32 link)
# /dev/spidev0.1  →  SPI controller 0, chip select 1  (another slave)
# /dev/spidev1.0  →  SPI controller 1, chip select 0

# Check which SPI controllers are in the device tree
cat /proc/device-tree/spi@3250000/status   # spi0
cat /proc/device-tree/spi@3270000/status   # spi1
```

**Permission gotcha:** `/dev/spidev*` is owned by `root:spidev` by default. Either add yourself to the
`spidev` group (`sudo usermod -aG spidev $USER`) or run as root — the latter is fine on a robot but a
bad habit for debugging.

---

### 1.2 Why jetson-io.py must be run first — the pinmux concept

**The analogy: a multi-function socket with a mode switch**

Imagine a wall socket that can be either a regular power outlet, a network jack, or a USB port — but it
can only be one thing at a time because all three options share the same wire pair inside the wall. A
physical dial on the fuse box sets which function is active.

Every pin on the Jetson's 40-pin GPIO header is exactly like this. Take pin 19 for example:

```
Pin 19 physically connects to ball T_B8 on the Tegra SoC.
That ball can be one of:
  - SPI1_MOSI     ← what we want
  - I2S4_DOUT     ← audio output
  - GPIO3_PCC.04  ← plain GPIO
  - DMIC3_CLK     ← digital mic
```

Only ONE of these functions can be active at a time. The selection is stored in the Tegra's **pinmux
registers** — hardware registers that live in the SoC and persist across reboots when stored in the boot
configuration.

`jetson-io.py` is NVIDIA's GUI tool that writes these registers correctly for you:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# Opens an ncurses menu
# Navigate to: "Configure 40-pin expansion header"
# Enable: SPI1 (CS0, CLK, MISO, MOSI)
# Save + reboot
```

**Why you MUST reboot:** `jetson-io.py` writes a new device tree overlay that the bootloader (CBoot)
applies when it initialises the SoC. The overlay configures the pinmux hardware before Linux even starts.
There is no way to reconfigure pinmux from Linux userspace after boot.

**The symptom if you skip it:** `/dev/spidev0.0` simply does not exist. `ls /dev/spidev*` returns nothing.
The error is not "permission denied" or "device busy" — the file literally is not there because the kernel
SPI driver was never told that SPI1 pins are wired for SPI.

```bash
# After jetson-io.py + reboot, verify:
ls -la /dev/spidev*
# Should show at least /dev/spidev0.0

# Also verify the hardware link before any Python code:
sudo spidev_test -D /dev/spidev0.0 -s 10000000 -p "AABBCCDD" -v
# This sends 0xAA 0xBB 0xCC 0xDD and shows what came back.
# If STM32 is in loopback mode: RX == TX. Otherwise RX = STM32's response.
```

---

### 1.3 The spi_ioc_transfer struct explained field by field

When Python calls `spi.xfer2(data)`, under the hood it fills in a C struct called `spi_ioc_transfer`
and hands it to the kernel via `ioctl`. Here is exactly what each field means:

```c
struct spi_ioc_transfer {
    __u64 tx_buf;        // pointer to bytes to send (MOSI)
    __u64 rx_buf;        // pointer to buffer for received bytes (MISO)
    __u32 len;           // number of bytes in this transfer
    __u32 speed_hz;      // SPI clock frequency for THIS transfer (0 = use device default)
    __u16 delay_usecs;   // microseconds to wait AFTER CS de-asserts before the next transfer
    __u8  bits_per_word; // bits per SPI word (0 = use device default, almost always 8)
    __u8  cs_change;     // if 1: de-assert CS after this transfer, re-assert for the next
    __u8  tx_nbits;      // for single/dual/quad SPI — use 0 (single) for standard SPI
    __u8  rx_nbits;      // same
    __u16 pad;           // alignment padding — set to 0
};
```

**tx_buf and rx_buf — the full-duplex rule:**

SPI is inherently full-duplex — every clock cycle pushes one bit out on MOSI AND shifts one bit in on
MISO simultaneously. This means every transfer sends AND receives the same number of bytes. If you don't
care about what comes back, you still allocate an rx_buf of the same length (just ignore its contents).
If you only want to receive, you fill tx_buf with 0x00 bytes (dummy bytes — the STM32 sees them but
ignores them if it's designed to).

**len — single field controls both directions:**

`len` is the total byte count. The kernel writes `len` bytes from tx_buf onto MOSI and simultaneously
fills rx_buf with `len` bytes from MISO. There's no separate tx_len and rx_len — it's always the same
because physical SPI exchanges one bit per clock on each wire simultaneously.

**speed_hz — per-transfer speed override:**

Setting `speed_hz = 0` uses the device's `max_speed_hz` (set via `spi.max_speed_hz = 10_000_000`).
You can override it per-transfer — useful if your protocol has a slow command phase followed by a fast
data phase. **The default kernel spidev speed is 500kHz** — 20x too slow for our 100Hz × 64-byte frame.
Always set this explicitly.

**delay_usecs — the gap between chained transfers:**

When you chain multiple `spi_ioc_transfer` structs in one `SPI_IOC_MESSAGE(n)` call, `delay_usecs`
controls the idle time between them. For most protocols: 0 (no gap, CS stays asserted). Useful when
talking to devices that need a "turnaround" time between a command send and a data receive.

**cs_change — who controls CS between chained transfers:**

- `cs_change = 0`: CS stays asserted for the entire chain (what you almost always want)
- `cs_change = 1`: CS de-asserts after this transfer, then re-asserts for the next

The STM32 sees a `cs_change = 1` as *two separate SPI transactions*. Use `cs_change = 0` for reading
a multi-phase protocol (header + payload) as one atomic operation.

---

### 1.4 SPI_IOC_MESSAGE(n) — why it's a single syscall that chains n transfers

**The analogy: a single FedEx waybill with multiple stops**

If you need to deliver packages to 3 locations, you could make 3 separate trips (inefficient) or put
all 3 packages on one truck with a single routing manifest (efficient). `SPI_IOC_MESSAGE(n)` is the
single-trip approach.

Normally, each `ioctl` call is one kernel context switch: your thread yields to the kernel, the kernel
does the work, returns. For a 3-phase SPI protocol (write command, wait, read response), doing 3 separate
`ioctl` calls costs 3 context switches and the OS can schedule other work between them — meaning another
thread could run between your "send command" and "read response." This breaks protocols that require CS
to stay asserted across phases.

`SPI_IOC_MESSAGE(n)` lets you pack n `spi_ioc_transfer` structs into ONE ioctl:

```python
import spidev
import array
import fcntl

# Build a 2-phase transfer: send 1-byte command, then receive 8 bytes,
# CS stays asserted throughout (cs_change = 0 on first struct)
#
# Python spidev library handles this for you via xfer2():
rx = spi.xfer2([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
# This is ONE ioctl(SPI_IOC_MESSAGE(1), ...) — single atomic transfer
```

For our use case: one `xfer2(64 bytes)` = one `SPI_IOC_MESSAGE(1)` = one ioctl = one context switch = 
one atomic SPI transaction with CS asserted for all 64 bytes. This is exactly what we want for reading
a full protobuf frame in one shot.

---

### 1.5 Why spidev ioctl blocks — your thread sleeps until the hardware is done

**The analogy: you ring the doorbell and wait at the door**

When you call `ioctl(fd, SPI_IOC_MESSAGE, ...)`, your calling thread is immediately **suspended by
the kernel** — it enters a sleep state and gives up the CPU. The kernel programs the DMA engine to
perform the SPI transfer. When the DMA completes, a hardware interrupt fires, the kernel's IRQ handler
runs, the data is copied to your buffer, and **your thread is woken up** — the ioctl returns.

From your code's perspective, the call blocks synchronously. From the OS's perspective, during the
~6.4µs it takes to clock 64 bytes at 10MHz, your process is not running at all, and the CPU can run
other processes.

**The practical consequence:**

At 10MHz SPI clock, a 64-byte frame takes:
```
64 bytes × 8 bits/byte = 512 bits
512 bits ÷ 10,000,000 bits/sec = 51.2 µs
```

Plus kernel scheduling overhead (2–5µs on standard Linux, <50µs on PREEMPT-RT): total ~55–60µs per
ioctl. Your thread blocks for that entire time.

This is fine when your thread's only job is SPI acquisition. It becomes a disaster if you put the
ioctl inside a ROS2 timer callback (see section 1.6).

---

### 1.6 The "ioctl in timer callback" anti-pattern — why it starves the executor

**The analogy: a receptionist who takes 50ms phone calls during a 10ms meeting**

The ROS2 executor is a single-threaded event loop. It manages ALL callbacks in your node:
- Timer callbacks (your 100Hz timer)
- Subscription callbacks (incoming messages)
- Service callbacks
- Action callbacks

When a timer fires, the executor calls your callback function. If that callback calls `xfer2()` —
which blocks for ~50–60ms waiting for SPI to complete — the executor is **frozen** for those 50ms.
No other callback can run. The timer itself cannot fire again. You've blocked the entire node.

**The result:** At 100Hz the timer should fire every 10ms. But your callback takes 50–60ms. The
executor schedules the next fire only AFTER the current callback returns. You get:

```
Timer fires at t=0ms  → callback blocks until t=55ms → returns
Timer fires at t=65ms → callback blocks until t=120ms → returns
Actual rate: 1000ms ÷ 65ms = ~15Hz instead of 100Hz
```

Even worse: the executor's timer resolution on a non-RT kernel has ~1-2ms jitter, adding further
deviation. In practice, learners see 83–87Hz and spend days tuning the callback instead of fixing
the architecture.

**Diagnosis:**

```bash
# Measure the actual rate you're getting
ros2 topic hz /imu/raw

# If it's 87Hz instead of 100Hz, the executor is stalled.
# The fix is NOT to make the SPI transfer faster —
# it's to move the SPI transfer OUT of the callback entirely.
```

---

### 1.7 Proper thread architecture — acquisition thread + queue + publish thread

**The analogy: a factory with a dedicated machine operator and a separate shipping clerk**

The correct architecture separates two concerns:

```
┌──────────────────────────────────────────────────────────┐
│  ACQUISITION THREAD (runs at SCHED_FIFO priority 90)     │
│  Tight loop: xfer2() → decode protobuf → queue.put()    │
│  No ROS2 involvement. Purely: grab data, enqueue it.    │
└─────────────────────────────┬────────────────────────────┘
                              │ queue.Queue (thread-safe)
                              ▼
┌──────────────────────────────────────────────────────────┐
│  ROS2 PUBLISH THREAD (timer callback, 100Hz)            │
│  queue.get_nowait() → fill Imu message → pub.publish()  │
│  If queue is empty: log warning, publish stale data.    │
└──────────────────────────────────────────────────────────┘
```

**Why this works:**

- The acquisition thread is a pure Python thread, blocked in `xfer2()` most of the time
- It never touches ROS2 internals — no GIL interaction with the executor
- The timer callback is now *very fast*: dequeue one item, fill a message struct, publish
- The executor is never blocked longer than a microsecond

**Key detail — queue depth:** Use `maxsize=5`. At 100Hz, each frame is 10ms apart. If the publisher
isn't consuming fast enough, you want to drop old frames (via `queue.get_nowait()` + full check), not
buffer them indefinitely. Stale sensor data is worse than no data for a Kalman filter.

**Python queue is thread-safe:** `queue.Queue` is protected internally by a mutex and condition
variable. You do NOT need additional locks around `queue.put()` and `queue.get()`.

---

### 1.8 PREEMPT-RT — what it is and what it changes

**The analogy: a hospital emergency room triage policy**

Standard Linux is like a hospital with a FIFO waiting room: you take a number and wait your turn
regardless of how sick you are. Your 100Hz SPI reader is sitting next to someone with a stubbed toe
waiting for their appointment.

PREEMPT-RT is like adding proper triage: real-time patients (your SPI reader) can immediately
preempt ongoing consultations for non-urgent cases. The key word is **fully preemptible kernel**.

In standard Linux, certain sections of kernel code run with interrupts **disabled** — nothing can
interrupt them, including your RT thread. These sections include spinlock holders, interrupt handlers,
and some scheduler code. These "blackout sections" can last 1–10ms on a loaded system.

PREEMPT-RT converts nearly all of these into **preemptible sections** by:
1. Converting interrupt handlers into high-priority kernel threads (they can be preempted by a
   higher-priority RT thread)
2. Converting spinlocks into mutexes (which support priority inheritance)
3. Making the scheduler itself preemptible

**Result:**

| Metric | Standard Linux | PREEMPT-RT |
|--------|---------------|------------|
| Worst-case scheduler latency | 1–10ms | 50–200µs |
| IRQ handler max latency | 500µs–5ms | 20–80µs |
| Can task miss 10ms deadline? | Yes, regularly | Extremely rare |

```bash
# Check if your Jetson kernel has PREEMPT-RT:
uname -r
# 5.15.148-rt76-tegra  ← the "rt" means PREEMPT-RT is compiled in

# Confirm it's active:
cat /sys/kernel/realtime
# 1 = active

# Measure worst-case latency baseline (100,000 samples, 4 threads):
sudo cyclictest -m -sp99 -t4 -l100000
# You want: Max latency < 100µs on a Jetson with good cooling
```

---

### 1.9 CPU isolation with isolcpus — the "reserved lane" analogy

**The analogy: a dedicated express lane on the motorway**

Imagine a 4-lane motorway. Normally, all cars (processes) use all lanes. During rush hour (high load),
your ambulance (RT thread) gets stuck behind a lorry. The OS's `isolcpus` kernel parameter is like
coning off CPU 3 as an **express lane that only ambulances can use.**

```
CPU 0: system processes, kernel workers, network IRQs
CPU 1: ROS2 nav stack, perception nodes
CPU 2: GPU driver threads, camera pipelines
CPU 3: ← ISOLATED — ONLY your SPI acquisition thread + its IRQ
```

**How to configure:**

1. Add to Jetson bootloader config:
```bash
sudo nano /boot/extlinux/extlinux.conf
# In the APPEND line, add:
# isolcpus=3 nohz_full=3 rcu_nocbs=3
```

2. Pin your SPI thread to the isolated core:
```python
import os
os.sched_setaffinity(0, {3})  # pin this process to CPU 3
```

3. Move the SPI controller's IRQ to CPU 3:
```bash
# Find which IRQ handles the SPI controller
cat /proc/interrupts | grep spi
# e.g., 103:    0    0    0    1234  SMP  103  tegra-spi

# Pin IRQ 103 to CPU 3:
echo 8 > /proc/irq/103/smp_affinity   # bitmask: 0b1000 = CPU 3
```

**Effect without isolation:**
- Your RT thread wakes up on CPU 3
- 0.5ms later, the kernel's RCU cleanup worker is scheduled on CPU 3
- Your thread gets preempted for 0.3ms
- This repeats every few hundred milliseconds: "spurious 0.3ms spikes"

**Effect with isolation:**
- Nothing except your thread and your IRQ handler ever runs on CPU 3
- Latency spikes from kernel housekeeping disappear

---

### 1.10 SCHED_FIFO vs SCHED_OTHER — why the default gives you 83Hz instead of 100Hz

**The analogy: ticketed vs priority boarding at an airline gate**

`SCHED_OTHER` (the default scheduling policy) is the **ticketed boarding** system: every process gets
a time slice (typically 1–4ms), and the scheduler rotates through all runnable processes fairly. Your
SPI thread waits its turn even if it's ready to run right now.

`SCHED_FIFO` is **priority boarding**: when your RT thread becomes runnable (e.g., after xfer2() 
returns), it immediately preempts any lower-priority `SCHED_OTHER` thread. No waiting for the next 
scheduler tick.

**At 100Hz, SCHED_OTHER causes timer drift:**

```
Intent:    fire every 10ms
Reality:   timer fires at 10ms, OS scheduler adds 1ms jitter → fires at 11ms
           next tick: 10ms + another 1ms jitter → fires at 22ms instead of 20ms
           ... drift accumulates → ~83–91Hz observed
```

The root cause: `create_wall_timer(10ms, cb)` in ROS2 means "at least 10ms between calls." If the
executor thread is waiting for a scheduler timeslice, that "at least" becomes "10ms + OS scheduling
latency."

**Setting SCHED_FIFO:**

```python
import ctypes
import ctypes.util

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

SCHED_FIFO = 1

class SchedParam(ctypes.Structure):
    _fields_ = [('sched_priority', ctypes.c_int)]

param = SchedParam(sched_priority=90)   # 90 = high RT priority (max 99)
ret = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
if ret != 0:
    raise OSError(ctypes.get_errno(), "sched_setscheduler failed — run as root?")
```

Or from the shell without changing code:
```bash
sudo chrt -f 90 python3 spi_reader.py    # SCHED_FIFO priority 90
```

**Priority 90 vs 99:** Priority 99 is reserved for kernel watchdog threads. Use 80–95 for your RT
acquisition. Standard ROS2 executors are at SCHED_OTHER (priority 0) so any positive SCHED_FIFO
priority will preempt them.

---

### 1.11 ROS2 QoS — RELIABLE vs BEST_EFFORT for sensor topics

**The analogy: registered letter vs postcard**

`RELIABLE` is like a registered letter: DDS (the ROS2 transport layer) stores the message, transmits
it, waits for an acknowledgement, and retransmits if the ack doesn't come. No message is ever lost.

`BEST_EFFORT` is like a postcard: you drop it in the postbox and move on. If the postal service
drops it, no retry — just use the next message.

**Why RELIABLE at 100Hz is dangerous:**

At 100Hz, a subscriber that falls behind by even 100ms has a backlog of 10 unread messages. With
RELIABLE, the DDS layer keeps them all queued and retransmits endlessly. The backlog grows. The
publisher sees backpressure. Eventually the queue overflows and `publish()` starts taking increasingly
long. Your 100Hz becomes 60Hz not because SPI is slow, but because DDS is trying to deliver stale
sensor data from 500ms ago.

**Why BEST_EFFORT is correct for sensors:**

If a subscriber misses frame 142, the next frame 143 arrives 10ms later. For an IMU or wheel encoder,
that's fine — use the freshest data. With `depth=5`, you keep the 5 most recent messages for any
subscriber that's slightly behind, but you never buffer hundreds of stale frames.

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data  # pre-built BEST_EFFORT profile

# Option 1: use the pre-built sensor profile (recommended)
pub = self.create_publisher(Imu, '/imu/raw', qos_profile_sensor_data)

# Option 2: explicit construction (same result)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
pub = self.create_publisher(Imu, '/imu/raw', sensor_qos)
```

**The silent failure — QoS mismatch:**

ROS2 silently drops the connection between a publisher and subscriber if their QoS policies are
incompatible. The rule: a RELIABLE subscriber will NOT receive messages from a BEST_EFFORT publisher.
The reverse is OK (BEST_EFFORT subscriber CAN receive from RELIABLE publisher). Neither side logs an
error. You see zero messages in `ros2 topic echo`, with no error anywhere.

```bash
# Detect mismatches — this shows every subscriber's actual QoS
ros2 topic info -v /imu/raw

# Look for:
# Publisher #1:  Reliability: BEST_EFFORT
# Subscription #1: Reliability: RELIABLE   ← MISMATCH — subscriber gets nothing
```

---

### 1.12 The timestamp problem — capture CLOCK_MONOTONIC_RAW immediately after ioctl

**The analogy: timestamping a lab sample at the centrifuge, not at the desk**

You have a blood sample. The centrifuge finishes at 14:03:22. The nurse walks to the desk, checks her
email, writes the label, and stamps "14:03:45" on the tube. That timestamp is wrong by 23 seconds.
A doctor looking at the lab result will calculate drug metabolism rates incorrectly.

Your IMU timestamp is the same problem. The xfer2() call returns with the IMU data at time T=0. If you
capture the timestamp 2ms later (after protobuf decode, after building the message struct, after calling
`self.get_clock().now()`), you stamp the data as T+2ms. The EKF integrates angular velocity using the
wrong dt and produces a ghost drift in the estimated orientation.

**Why 2ms matters at 100Hz:**

```
IMU publish rate: 100Hz → one frame every 10ms
Angular velocity: 1.0 rad/s (slow rotation)
Timestamp error:  2ms = 0.002s
Position error from integration: 1.0 rad/s × 0.002s = 0.002 rad = 0.11 degrees
Over 100 frames (1 second): this accumulates, not averages out
```

The EKF will diagnose this as "noise," suggest you reduce process noise covariance, which makes the
filter trust the IMU more, which makes the drift worse. Learners spend 3 days tuning EKF parameters
when the real fix is one line of code.

**The correct pattern:**

```python
import ctypes
import ctypes.util

# Load libc for clock_gettime
librt = ctypes.CDLL(ctypes.util.find_library('rt'), use_errno=True)

class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

CLOCK_MONOTONIC_RAW = 4  # Linux constant — immune to NTP adjustments

def get_monotonic_raw_ns() -> int:
    """Return nanoseconds since boot (CLOCK_MONOTONIC_RAW)."""
    ts = Timespec()
    librt.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec

# In the acquisition thread:
rx_bytes = spi.xfer2([0x00] * FRAME_SIZE)       # blocks until SPI complete
capture_ns = get_monotonic_raw_ns()              # ← IMMEDIATELY after xfer2 returns
frame = SensorFrame()
frame.ParseFromString(bytes(rx_bytes[3:]))       # decode AFTER capturing the timestamp
# ... enqueue (frame, capture_ns)
```

**Why `CLOCK_MONOTONIC_RAW` not `get_clock().now()`:**

`get_clock().now()` in rclpy uses `RCL_SYSTEM_TIME` which is corrected by NTP in real-time. NTP can
adjust the clock backwards or jump it forward. `CLOCK_MONOTONIC_RAW` is immune to NTP adjustments and
never goes backwards — exactly what you want for a Kalman filter's dt calculation.

---

### 1.13 robot_localization covariance matrix — why zeros crash the EKF

**The analogy: a weather forecaster who refuses to admit uncertainty**

A Kalman filter is fundamentally a **weighted average** between a prediction and a measurement. The
weights come from:
- Prediction covariance (how uncertain is our model?)
- Measurement covariance (how uncertain is this sensor reading?)

The covariance matrix in a `sensor_msgs/Imu` message is a 3×3 matrix flattened into a 9-element
array. The diagonal elements mean:

```
linear_acceleration_covariance:
  [σ²_ax, 0,     0,
   0,     σ²_ay, 0,
   0,     0,     σ²_az]

Where σ²_ax = variance of x-acceleration measurement in (m/s²)²
```

**What zero means:** `σ² = 0` means "this measurement has zero uncertainty — it is infinitely
accurate." The EKF's observation gain equation is:
```
K = P × H^T × (H × P × H^T + R)^(-1)
```
When R (measurement noise covariance) has zeros on the diagonal, the matrix inversion becomes
numerically unstable, the Kalman gain explodes, and the filter diverges — often to NaN.

**The two conventions and the special value -1.0:**

```python
# Convention 1: set -1.0 in position [0] = "I don't know the covariance"
#               robot_localization will use whatever you set in its YAML config instead
msg.linear_acceleration_covariance[0] = -1.0

# Convention 2: set actual calibrated values (better if you have them)
msg.linear_acceleration_covariance = [
    0.01, 0.0,  0.0,   # σ²_ax = 0.01 (m/s²)²
    0.0,  0.01, 0.0,   # σ²_ay
    0.0,  0.0,  0.01,  # σ²_az
]

# NEVER do this — zero covariance = EKF divergence:
msg.linear_acceleration_covariance = [0.0] * 9  # ← this will crash your EKF
```

**robot_localization YAML — what the config covariance means:**

In the `robot_localization` config YAML, you set `process_noise_covariance` (how much the EKF trusts
its own motion model) and `initial_estimate_covariance` (how uncertain is the initial state). These
are completely separate from the per-message measurement covariance. Confusing them is one of the
most common EKF tuning mistakes.

---

### 1.14 TF2 frame tree — map → odom → base_link → imu_frame

**The analogy: a chain of address changes**

Your robot starts at the origin of the warehouse (map frame). As it moves, it accumulates a position
estimate (odom frame). Its body (base_link frame) has sensors attached at known offsets
(imu_frame, laser_frame).

TF2 is the ROS2 library that manages these relationships and lets any node ask "where is my IMU
in the map frame RIGHT NOW?" without having to manually chain transformations.

```
map                    ← absolute coordinate system; set by AMCL or GPS
  └── odom             ← robot's dead-reckoning position; published by EKF
        └── base_link  ← robot body center; the reference for everything on the robot
              ├── imu_link      ← where the IMU sits, with its mounting orientation
              ├── laser_link    ← where the LiDAR sits
              └── camera_link   ← where the camera sits
```

**What each edge represents physically:**

- `map → odom`: the correction applied by the localization system (AMCL, GPS). This edge fills in
  where the robot is in the global map. When AMCL updates, it adjusts this edge. This edge can jump
  (that's fine — it corrects accumulated drift).

- `odom → base_link`: the EKF's integrated estimate of how far the robot has moved since boot.
  This edge is continuous and smooth (never jumps). Published by `robot_localization`.

- `base_link → imu_link`: the static physical offset between the robot's body center and where the
  IMU is mounted. This never changes during a run — published once by a static transform broadcaster.

**Why the frame_id in your IMU message MUST be correct:**

When you set `msg.header.frame_id = 'imu_link'`, you're telling TF2 "this data is measured in the
imu_link frame." `robot_localization` uses TF2 to look up the transform from `imu_link` to
`base_link` and rotate your accelerations/angular velocities into the body frame before fusing.

If you write `'imu'` instead of `'imu_link'`, TF2 searches for a frame named `'imu'` in the tree,
finds nothing, and `robot_localization` discards the message silently. The EKF runs on odometry
alone — your IMU is ignored with no error message.

```bash
# Verify your TF tree is complete:
ros2 run tf2_tools view_frames
evince frames.pdf   # opens a visualization

# Live TF lookup test:
ros2 run tf2_ros tf2_echo base_link imu_link
# Should print: Translation + Rotation of imu_link relative to base_link
```

---

### 1.15 The irq_balance gotcha — spurious 2ms spikes despite RT setup

**The analogy: a well-meaning traffic manager who keeps changing your lane**

`irqbalance` is a Linux service that periodically re-examines which CPU handles which hardware
interrupt, and migrates IRQs to balance CPU load. This is a good idea for busy servers running many
processes. For a real-time acquisition thread, it's a slow-motion disaster.

Here is the exact failure sequence:

```
t=0ms:      SPI acquisition thread on CPU3, SPI IRQ on CPU3 → perfect
t=3000ms:   irqbalance runs, migrates SPI IRQ from CPU3 to CPU1
t=3000ms+:  SPI ioctl starts on CPU3, DMA finishes, IRQ fires on CPU1
            CPU1 runs the IRQ handler, updates the kernel SPI wait queue
            CPU3 is notified via IPI (inter-processor interrupt) to wake up
            IPI is not instantaneous: ~0.5-2ms additional latency
t=3002ms:   xfer2() returns, 2ms late
```

The spike is intermittent (every few hundred seconds when irqbalance re-balances), disappears on
restart (irqbalance randomizes assignments), and varies with system load. This makes it look like
random noise in your timing measurements.

**The fix:**

```bash
# Permanent: disable irqbalance service
sudo systemctl disable irqbalance
sudo systemctl stop irqbalance

# Then manually pin the SPI IRQ to your isolated core:
# Find the SPI IRQ number:
cat /proc/interrupts | grep spi
# Pin to CPU 3 (bitmask 0b1000 = 8):
echo 8 > /proc/irq/<SPI_IRQ_NUMBER>/smp_affinity
```

**Verify it worked:**

After pinning, run `cyclictest` while simultaneously doing SPI transfers and watch that worst-case
latency stays flat instead of occasionally spiking. A spike of >500µs every few minutes that disappears
after `systemctl stop irqbalance` is the definitive diagnosis.

---

## PART 2 — Annotated Code Reference

---

### 2.1 spidev Python setup — open, configure, xfer2

```python
import spidev

# ── Step 1: Create the spidev instance ──────────────────────────────────────
spi = spidev.SpiDev()
# spidev.SpiDev() allocates the wrapper object. No kernel interaction yet.

# ── Step 2: Open the device ─────────────────────────────────────────────────
spi.open(0, 0)
# Opens /dev/spidev0.0 — bus=0, chip_select=0
# This is a blocking open() syscall. Fails with PermissionError if user is
# not in the 'spidev' group. Fails with FileNotFoundError if jetson-io.py
# was not run + reboot.

# ── Step 3: Set SPI clock speed ──────────────────────────────────────────────
spi.max_speed_hz = 10_000_000
# Sets the maximum clock frequency: 10 MHz
# CRITICAL: the kernel default is 500_000 (500kHz) — 20× too slow for 100Hz × 64 bytes.
# At 500kHz: 64 bytes × 8 bits = 512 bits @ 500kHz = 1.024ms per transfer
# At 10MHz:  512 bits @ 10MHz  = 51.2µs per transfer → fits in 10ms budget

# ── Step 4: Set SPI mode ─────────────────────────────────────────────────────
spi.mode = 0b00
# SPI mode encodes two bits:
#   CPOL (bit 1): clock polarity — 0 = idle LOW, 1 = idle HIGH
#   CPHA (bit 0): clock phase   — 0 = sample on LEADING edge, 1 = sample on TRAILING edge
#
# Mode 0 (CPOL=0, CPHA=0): clock idles LOW, data sampled on rising edge
# Mode 1 (CPOL=0, CPHA=1): clock idles LOW, data sampled on falling edge
# Mode 2 (CPOL=1, CPHA=0): clock idles HIGH, data sampled on falling edge
# Mode 3 (CPOL=1, CPHA=1): clock idles HIGH, data sampled on rising edge
#
# STM32 SPI_CPOL_Low + SPI_CPHA_1Edge = Mode 0.
# If your mode is WRONG you'll receive bytes that are consistently shifted
# — logic analyzer shows wrong byte values; try modes 1,2,3 if rx is garbage.

spi.bits_per_word = 8
# Almost always 8. The SPI protocol can support 4-16 bit words,
# but our STM32 frame uses byte-sized units. Don't change this.

spi.no_cs = False
# False = let spidev assert/deassert CS automatically around each transfer.
# True  = you control CS manually via a GPIO. Rarely needed.

# ── Step 5: Transfer ─────────────────────────────────────────────────────────
FRAME_SIZE = 64    # must match STM32 frame size

tx_dummy = [0x00] * FRAME_SIZE
# Dummy bytes — the STM32 ignores what the Jetson sends; it just waits for
# CLK pulses to clock out its pre-built frame. We send zeros to make debugging
# easier (0x00 on MOSI is easy to filter out in a logic analyzer capture).

rx = spi.xfer2(tx_dummy)
# xfer2(): CS asserted at start, held throughout, de-asserted at end.
# Full-duplex: sends tx_dummy on MOSI, fills rx with MISO bytes simultaneously.
# WHY xfer2 not xfer: xfer() would de-assert CS between each BYTE.
# xfer2() keeps CS asserted for the entire 64-byte transfer → one transaction.
# The STM32 relies on CS staying low to know it's part of one frame.

received_bytes = bytes(rx)   # convert list-of-ints to bytes for protobuf

spi.close()   # release the file descriptor; good practice on shutdown
```

---

### 2.2 The correct acquisition thread — SCHED_FIFO, mlockall, tight loop

```python
import threading
import queue
import ctypes
import ctypes.util
import spidev
import time

# ── Real-time setup helpers ──────────────────────────────────────────────────

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
librt = ctypes.CDLL(ctypes.util.find_library('rt'), use_errno=True)

SCHED_FIFO = 1
MCL_CURRENT = 1
MCL_FUTURE = 2

class SchedParam(ctypes.Structure):
    _fields_ = [('sched_priority', ctypes.c_int)]

class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

CLOCK_MONOTONIC_RAW = 4


def _get_time_ns() -> int:
    """Read CLOCK_MONOTONIC_RAW in nanoseconds. Must be called immediately after xfer2()."""
    ts = Timespec()
    librt.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def _set_realtime(priority: int = 90) -> None:
    """Promote current thread to SCHED_FIFO. Must be called from the thread itself."""
    param = SchedParam(sched_priority=priority)
    ret = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
    # 0 = current thread/process
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"sched_setscheduler failed (errno={err}): run as root")

    # Lock all pages into RAM — prevents page faults delaying the RT thread.
    # A page fault takes 0.1–1ms; unacceptable for a 10ms deadline.
    # MCL_CURRENT: lock pages already in memory
    # MCL_FUTURE:  lock pages that will be mapped in future
    ret = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mlockall failed (errno={err})")


# ── The acquisition thread function ─────────────────────────────────────────

FRAME_SIZE = 64          # must match STM32 DMA frame size
QUEUE_MAX  = 5           # drop old frames rather than buffer unboundedly

def _acquisition_thread(
    data_queue: queue.Queue,
    stop_event: threading.Event,
    spi_bus: int = 0,
    spi_cs: int = 0,
) -> None:
    """
    Runs at SCHED_FIFO priority 90 on the isolated CPU core.
    Does NOTHING except:
      1. Call xfer2() to grab a frame
      2. Capture the timestamp immediately
      3. Push (raw_bytes, timestamp_ns) to data_queue
    No logging, no ROS2, no protobuf decode in this thread.
    Logging from an RT thread costs ~0.5ms if the log buffer is full.
    """
    _set_realtime(priority=90)
    # Pin this thread to CPU3 (the isolated core)
    import os
    os.sched_setaffinity(0, {3})

    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_cs)
    spi.max_speed_hz = 10_000_000    # 10 MHz
    spi.mode = 0b00                  # CPOL=0 CPHA=0
    spi.bits_per_word = 8
    spi.no_cs = False

    tx_dummy = [0x00] * FRAME_SIZE   # allocated once — no malloc in the hot loop

    while not stop_event.is_set():
        raw = spi.xfer2(tx_dummy)           # blocking SPI transfer
        ts_ns = _get_time_ns()              # IMMEDIATELY capture timestamp — do not delay

        payload = bytes(raw)

        if data_queue.full():
            try:
                data_queue.get_nowait()     # drop oldest item to make room
            except queue.Empty:
                pass

        try:
            data_queue.put_nowait((payload, ts_ns))
        except queue.Full:
            pass                            # extremely rare race condition — skip frame

    spi.close()


def start_acquisition(spi_bus=0, spi_cs=0) -> tuple[queue.Queue, threading.Event]:
    """Start the RT acquisition thread. Returns (queue, stop_event)."""
    data_q = queue.Queue(maxsize=QUEUE_MAX)
    stop_ev = threading.Event()
    t = threading.Thread(
        target=_acquisition_thread,
        args=(data_q, stop_ev, spi_bus, spi_cs),
        daemon=True,      # thread dies automatically when main process exits
        name='spi_acq',
    )
    t.start()
    return data_q, stop_ev
```

---

### 2.3 ROS2 Python publisher node — proper structure with SensorDataQoS, correct timestamp

```python
#!/usr/bin/env python3
"""
spi_bridge_node.py — STM32 SPI to ROS2 bridge

Architecture:
  - acquisition thread (SCHED_FIFO, CPU3): reads SPI at 100Hz, timestamps immediately
  - ROS2 executor thread: dequeues frames, publishes to /imu/raw + /wheel_speed
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data   # pre-built BEST_EFFORT, depth=10
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped

from sensor_frame_pb2 import SensorFrame        # generated from sensor_frame.proto
from .spi_acquisition import start_acquisition  # the thread from section 2.2

import queue


def _ns_to_ros_time(ns: int) -> Time:
    """Convert nanoseconds since epoch to ROS2 Time message."""
    msg = Time()
    msg.sec     = ns // 1_000_000_000
    msg.nanosec = ns  % 1_000_000_000
    return msg


class SpiBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('spi_bridge')

        # ── Declare parameters ────────────────────────────────────────────────
        self.declare_parameter('spi_bus', 0)         # /dev/spidev<bus>.<cs>
        self.declare_parameter('spi_cs',  0)
        self.declare_parameter('imu_frame_id', 'imu_link')   # TF frame for IMU
        self.declare_parameter('base_frame_id', 'base_link')

        bus       = self.get_parameter('spi_bus').value
        cs        = self.get_parameter('spi_cs').value
        self._imu_frame  = self.get_parameter('imu_frame_id').value
        self._base_frame = self.get_parameter('base_frame_id').value

        # ── Publishers ────────────────────────────────────────────────────────
        # qos_profile_sensor_data = BEST_EFFORT + VOLATILE + KEEP_LAST(10)
        # Use BEST_EFFORT for ALL sensor topics at 100Hz — never RELIABLE.
        self._imu_pub   = self.create_publisher(Imu,          '/imu/raw',     qos_profile_sensor_data)
        self._wheel_pub = self.create_publisher(TwistStamped, '/wheel_speed', qos_profile_sensor_data)

        # ── Start RT acquisition thread ───────────────────────────────────────
        self._data_q, self._stop_ev = start_acquisition(bus, cs)

        # ── Statistics ───────────────────────────────────────────────────────
        self._pub_ok  = 0
        self._pub_err = 0

        # ── 100Hz publish timer ───────────────────────────────────────────────
        # This callback is now VERY fast: dequeue + fill message + publish.
        # No SPI blocking. No protobuf decode (do that inside _publish).
        self.create_timer(0.010, self._publish_cb)

        # ── Periodic stats log ────────────────────────────────────────────────
        self.create_timer(5.0, self._stats_cb)

        self.get_logger().info(
            f'spi_bridge started: /dev/spidev{bus}.{cs} @ 10MHz, '
            f'imu_frame={self._imu_frame}'
        )

    def _publish_cb(self) -> None:
        """Called every 10ms by the executor. Dequeues ONE frame and publishes it."""
        try:
            raw_bytes, ts_ns = self._data_q.get_nowait()
        except queue.Empty:
            # SPI acquisition hasn't produced a frame yet (startup transient)
            # or the acquisition thread is behind — log at warn, not error
            self.get_logger().warn('publish_cb: queue empty')
            return

        try:
            frame = SensorFrame()
            frame.ParseFromString(raw_bytes)
            self._publish_frame(frame, ts_ns)
            self._pub_ok += 1
        except Exception as exc:
            self.get_logger().error(f'Frame decode error: {exc}')
            self._pub_err += 1

    def _publish_frame(self, frame: SensorFrame, ts_ns: int) -> None:
        """Fill and publish ROS2 messages from a decoded SensorFrame."""
        ros_stamp = _ns_to_ros_time(ts_ns)   # timestamp of the ACTUAL SPI transfer

        if frame.HasField('imu'):
            msg = Imu()
            msg.header.stamp    = ros_stamp       # captured right after xfer2() returned
            msg.header.frame_id = self._imu_frame # must match your TF tree

            msg.linear_acceleration.x = frame.imu.accel_x
            msg.linear_acceleration.y = frame.imu.accel_y
            msg.linear_acceleration.z = frame.imu.accel_z
            msg.angular_velocity.x    = frame.imu.gyro_x
            msg.angular_velocity.y    = frame.imu.gyro_y
            msg.angular_velocity.z    = frame.imu.gyro_z

            # -1.0 in position [0] = "covariance unknown; use robot_localization YAML config"
            # DO NOT leave all zeros — that means "infinitely accurate" → EKF diverges
            msg.linear_acceleration_covariance[0] = -1.0
            msg.angular_velocity_covariance[0]    = -1.0
            msg.orientation_covariance[0]         = -1.0  # IMU has no mag, no orientation

            self._imu_pub.publish(msg)

        if frame.HasField('wheel'):
            msg = TwistStamped()
            msg.header.stamp    = ros_stamp
            msg.header.frame_id = self._base_frame
            vl = frame.wheel.speed_left
            vr = frame.wheel.speed_right
            msg.twist.linear.x  = (vl + vr) / 2.0     # average forward speed
            msg.twist.angular.z = (vr - vl) / 0.30     # angular from differential (0.30m wheelbase)
            self._wheel_pub.publish(msg)

    def _stats_cb(self) -> None:
        total = self._pub_ok + self._pub_err
        rate  = self._pub_ok / max(total, 1) * 100.0
        self.get_logger().info(
            f'spi_bridge: ok={self._pub_ok} err={self._pub_err} '
            f'queue_size={self._data_q.qsize()} success={rate:.1f}%'
        )

    def destroy_node(self) -> None:
        self._stop_ev.set()     # signal acquisition thread to exit
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpiBridgeNode()
    try:
        rclpy.spin(node)        # runs executor; calls _publish_cb every 10ms
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

### 2.4 The WRONG publisher vs the CORRECT publisher — side by side

```python
# ════════════════════════════════════════════════════════════════════════════
# ❌ WRONG: ioctl (xfer2) inside the timer callback
# ════════════════════════════════════════════════════════════════════════════

class WrongBridgeNode(Node):
    def __init__(self):
        super().__init__('wrong_bridge')
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 10_000_000
        self.spi.mode = 0
        self.pub = self.create_publisher(Imu, '/imu/raw', qos_profile_sensor_data)

        # ← Timer callback does EVERYTHING: SPI + decode + publish
        self.create_timer(0.010, self._cb)

    def _cb(self):
        # ← This xfer2() BLOCKS for ~55µs, holding the executor hostage
        # The executor cannot fire ANY other callback until this returns.
        raw = self.spi.xfer2([0x00] * 64)

        # ← Timestamp captured AFTER the transfer — already 0.1ms late minimum
        # The time from xfer2() return to get_clock().now() adds processing jitter.
        stamp = self.get_clock().now().to_msg()

        frame = SensorFrame()
        frame.ParseFromString(bytes(raw))   # decode here adds another 0.1ms

        msg = Imu()
        msg.header.stamp = stamp            # wrong! already stale
        msg.linear_acceleration.x = frame.imu.accel_x
        self.pub.publish(msg)
        # Total callback duration: ~200µs minimum, up to 3ms on a loaded system
        # Actual publish rate: ~87Hz instead of 100Hz
        # EKF sees 2ms timestamp error → phantom orientation drift


# ════════════════════════════════════════════════════════════════════════════
# ✅ CORRECT: dedicated RT acquisition thread + fast publish callback
# ════════════════════════════════════════════════════════════════════════════

class CorrectBridgeNode(Node):
    def __init__(self):
        super().__init__('correct_bridge')
        self.pub = self.create_publisher(Imu, '/imu/raw', qos_profile_sensor_data)

        # ← Acquisition happens in a SEPARATE Python thread at SCHED_FIFO priority.
        # xfer2() blocking does NOT affect the executor.
        self._q, self._stop = start_acquisition(spi_bus=0, spi_cs=0)

        # ← Timer callback is now <5µs: dequeue + message fill + publish
        self.create_timer(0.010, self._cb)

    def _cb(self):
        try:
            raw, ts_ns = self._q.get_nowait()   # non-blocking dequeue
        except queue.Empty:
            return                               # acquisition thread not ready yet

        # ← Timestamp from the acquisition thread: captured IMMEDIATELY after xfer2
        msg = Imu()
        msg.header.stamp    = _ns_to_ros_time(ts_ns)   # accurate to <10µs
        msg.header.frame_id = 'imu_link'
        frame = SensorFrame()
        frame.ParseFromString(raw)
        msg.linear_acceleration.x = frame.imu.accel_x
        msg.linear_acceleration_covariance[0] = -1.0   # not zero!
        self.pub.publish(msg)
        # Total callback duration: ~5µs typical; executor never stalled
        # Actual publish rate: 100.0Hz ± 0.3Hz
        # EKF timestamp error: <50µs
```

**Key differences at a glance:**

| | WRONG (ioctl in callback) | CORRECT (thread + queue) |
|--|--|--|
| Executor blocked during xfer2? | Yes — ~55µs | No |
| Actual publish rate | ~83–87Hz | 99.5–100.5Hz |
| Timestamp accuracy | 0.5–3ms error | <50µs error |
| EKF behaviour | Phantom drift | Converges correctly |
| Other callbacks delayed? | Yes | No |

---

### 2.5 cyclictest — measuring your RT latency baseline

`cyclictest` is the standard tool for measuring how late a real-time task wakes up compared to its
scheduled wakeup time. Run this BEFORE starting any node to confirm your RT baseline is acceptable.

```bash
# Basic command:
sudo cyclictest -m -sp99 -t4 -l100000

# Flag breakdown:
#   -m          : mlockall — prevent page faults during the test itself
#   -s          : show summary at end
#   -p99        : SCHED_FIFO priority 99 — maximum RT priority to expose worst-case kernel latency
#   -t4         : run 4 threads (one per core — stresses all CPUs simultaneously)
#   -l100000    : run 100,000 iterations per thread (takes ~100 seconds at -i 1000µs default)

# Example output:
# T: 0 (14423) P:99 I:1000 C: 100000 Min:    12 Act:   14 Avg:   18 Max:    67
# T: 1 (14424) P:99 I:1000 C: 100000 Min:    11 Act:   13 Avg:   17 Max:    72
# T: 2 (14425) P:99 I:1000 C: 100000 Min:    13 Act:   15 Avg:   19 Max:    81
# T: 3 (14426) P:99 I:1000 C: 100000 Min:    12 Act:   14 Avg:   18 Max:    69

# Column meanings:
#   T:           thread index
#   (14423):     PID
#   P:99:        priority
#   I:1000:      interval in microseconds (1ms between wakeups)
#   C: 100000:   count of completed iterations
#   Min: 12:     best-case wakeup latency = 12µs
#   Act: 14:     most recent iteration latency
#   Avg: 18:     average latency over all iterations
#   Max: 67:     WORST-CASE latency = 67µs ← this is the number you care about

# Acceptance criteria for 100Hz SPI acquisition:
#   Max < 100µs  → excellent, 10ms budget with <1% consumed by latency
#   Max < 500µs  → acceptable for 100Hz
#   Max > 1000µs → investigate: irqbalance? thermal throttling? wrong kernel?

# Run while your ROS2 node is active (stresses the system realistically):
sudo cyclictest -m -sp90 -t1 -l100000 &
ros2 run spi_bridge spi_bridge &
# Compare Max values with and without the ROS2 node running
```

---

### 2.6 strace -T — measuring actual ioctl duration per call

```bash
# Trace only ioctl calls, show wall-clock time per call:
sudo strace -T -e trace=ioctl python3 spi_reader.py 2>&1 | grep SPI

# Example output (one line per ioctl call):
# ioctl(3, SPI_IOC_MESSAGE(1), 0x7ff8b24000) = 1 <0.000058>
# ioctl(3, SPI_IOC_MESSAGE(1), 0x7ff8b24000) = 1 <0.000061>
# ioctl(3, SPI_IOC_MESSAGE(1), 0x7ff8b24000) = 1 <0.000210>  ← spike!
# ioctl(3, SPI_IOC_MESSAGE(1), 0x7ff8b24000) = 1 <0.000059>

# The number in <> is WALL-CLOCK time for that single ioctl call, in seconds.
# 0.000058 = 58µs (normal for 64-byte frame at 10MHz)
# 0.000210 = 210µs (spike — likely IRQ migration or cache miss)

# The ioctl number SPI_IOC_MESSAGE(1) breaks down as:
#   SPI_IOC_MESSAGE(n): ioctl that chains n spi_ioc_transfer structs
#   The (1) means one transfer struct — our single 64-byte transfer
#   Return value "= 1" means: success, 1 transfer completed

# Useful combination: watch for spikes AND confirm nominal duration:
sudo strace -T -e trace=ioctl python3 spi_reader.py 2>&1 \
    | grep SPI \
    | awk '{print $NF}' \
    | tr -d '<>' \
    | awk '{if ($1+0 > 0.0001) print "SPIKE: " $1}'
    # prints only ioctls taking >100µs
```

---

### 2.7 robot_localization YAML config — covariance matrix annotation

```yaml
# ekf_config.yaml — robot_localization EKF node configuration
# Located: ~/ros2_ws/src/my_robot/config/ekf_config.yaml

ekf_node:
  ros__parameters:
    # ── Coordinate frames ─────────────────────────────────────────────────────
    world_frame: odom           # frame published by this EKF (map or odom)
    odom_frame: odom            # the EKF publishes the odom→base_link transform
    base_link_frame: base_link

    # ── Input sensors ────────────────────────────────────────────────────────
    imu0: /imu/raw              # Topic for IMU input
    imu0_config:                # 15 booleans: which fields to fuse
      # Format: [roll, pitch, yaw, vroll, vpitch, vyaw, ax, ay, az, vx, vy, vz, x, y, z]
      # False = ignore that field from this sensor
      - [false, false, false,   # roll, pitch, yaw    — IMU has no magnetometer → no absolute heading
         true,  true,  true,    # vroll, vpitch, vyaw — angular velocities from gyro ← fuse these
         true,  true,  true,    # ax, ay, az          — linear accelerations from accel ← fuse these
         false, false, false,   # vx, vy, vz          — velocities: not from IMU
         false, false, false]   # x, y, z             — position: not from IMU

    imu0_differential: false    # false: use absolute values; true: use incremental changes
    imu0_relative:     false    # false: use sensor's frame; true: interpret w.r.t. robot start pose
    imu0_queue_size:   10       # internal queue; increase if messages arrive in bursts

    odom0: /wheel_speed         # Topic for wheel odometry
    odom0_config:
      - [false, false, false,
         false, false, false,
         false, false, false,
         true,  false, true,    # vx (forward velocity) and vyaw (from differential drive)
         false, false, false]

    # ── Process noise covariance (Q matrix) ─────────────────────────────────
    # How much we trust our OWN MOTION MODEL between measurements.
    # Larger value = "our model is uncertain" → EKF weights sensor data more heavily.
    # 15×15 matrix (flattened). Elements correspond to:
    # [roll, pitch, yaw, vroll, vpitch, vyaw, ax, ay, az, vx, vy, vz, x, y, z]
    # Diagonal entries matter; off-diagonal usually 0.
    process_noise_covariance:   # Q matrix, 15×15 flattened to 225 values
      - [0.05, 0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0.05, 0,    0,    0,    0,    0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0.06, 0,    0,    0,    0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0.03, 0,    0,    0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0.03, 0,    0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0.06, 0,     0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0.025, 0,     0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0.025, 0,     0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0.04,  0,    0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0.01, 0,    0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0.01, 0,    0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0.02, 0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,    0.1,  0,    0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,    0,    0.1,  0,
         0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,    0,    0,    0.15]
    # ↑ The [0,0] element (0.05) = process noise for roll: how much roll can change per second²
    # ↑ The [6,6] element (0.025) = process noise for ax: accelerometer model uncertainty
    # Rule of thumb: start with these values; if EKF reacts too slowly → increase;
    #                if EKF is noisy → decrease

    # ── Initial state covariance (P₀ matrix) ─────────────────────────────────
    # How uncertain we are about the robot's STARTING state.
    # High values = "I have no idea where I am at boot" (usually correct)
    initial_estimate_covariance:  # 15×15, same structure as process_noise_covariance
      - [1e-9, 0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    1e-9, 0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    1e-9, 0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    1e-9, 0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    1e-9, 0,    0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    1e-9, 0,    0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    1e-9, 0,    0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    1e-9, 0,    0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    1e-9, 0,     0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    1e-9,  0,     0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     1e-9,  0,     0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     1e-9,  0,    0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0.5,  0,    0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0.5,  0,
         0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     0,     0,     0,    0,    0.5]
    # ↑ Small values for rotations/velocities (we start at rest = well-known)
    # ↑ 0.5 for x, y, z position (we might not know where we are yet)
```

---

### 2.8 TF2 static transform broadcaster for imu_link

```python
#!/usr/bin/env python3
"""
Publish the static transform between base_link and imu_link.
This runs once at startup and keeps the transform alive via /tf_static.
"""

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class ImuTfPublisher(Node):
    def __init__(self) -> None:
        super().__init__('imu_tf_publisher')
        self._broadcaster = StaticTransformBroadcaster(self)

        tf = TransformStamped()

        # ── Header ────────────────────────────────────────────────────────────
        tf.header.stamp    = self.get_clock().now().to_msg()
        tf.header.frame_id = 'base_link'     # parent frame
        tf.child_frame_id  = 'imu_link'      # child frame

        # ── Translation: where is the IMU relative to base_link center? ──────
        # Positive x = forward, positive y = left, positive z = up
        # Example: IMU is 5cm forward, 0 lateral, 3cm above base center
        tf.transform.translation.x = 0.05    # 5cm forward
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.03    # 3cm above base_link origin

        # ── Rotation: how is the IMU mounted? ─────────────────────────────────
        # If IMU x-axis aligns with robot forward and z-axis points up:
        # → no rotation needed → identity quaternion
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0       # identity quaternion = no rotation

        # If IMU is mounted rotated 90° around Z (IMU x = robot left):
        # tf.transform.rotation.z = math.sin(math.pi / 4)  # sin(45°)
        # tf.transform.rotation.w = math.cos(math.pi / 4)  # cos(45°)

        # StaticTransformBroadcaster.sendTransform() publishes to /tf_static
        # /tf_static is LATCHED — new subscribers get the transform immediately,
        # even if they subscribe after this node has already published.
        self._broadcaster.sendTransform(tf)
        self.get_logger().info(
            f'Published static TF: {tf.header.frame_id} → {tf.child_frame_id}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuTfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Verify the transform was published:**
```bash
# List all known TF frames:
ros2 run tf2_tools view_frames && evince frames.pdf

# Query the specific transform:
ros2 run tf2_ros tf2_echo base_link imu_link
# Should print translation + rotation without error

# If you see "lookup would require extrapolation into the future":
# The TF is not being published. Check your node is running.

# If you see "frame does not exist":
# Common causes:
#   1. frame_id typo ("imu" vs "imu_link")
#   2. Static broadcaster node not running
#   3. /tf_static not being published (check ros2 topic echo /tf_static)
```

---

## PART 3 — Gotcha Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| `/dev/spidev0.0` does not exist | `jetson-io.py` not run, or run but no reboot | `ls /dev/spidev*` returns empty. Check `dmesg | grep spi`. No spidev kernel module loaded for that bus. | Run `sudo /opt/nvidia/jetson-io/jetson-io.py`, configure SPI1, save, then **reboot** the Jetson. |
| `ioctl` hangs forever, node freezes | STM32 reset mid-transfer with CS still asserted | Node stops responding. `strace` shows ioctl blocking with no return. The kernel SPI driver has no timeout. | Run SPI ioctl in a dedicated thread with a watchdog timer. Detect STM32 reset via a GPIO "heartbeat" pin. |
| Default speed too slow — can't hit 100Hz | `max_speed_hz` not set, defaults to 500kHz kernel default | At 500kHz, 64-byte frame takes 1.024ms. At 100Hz you need <10ms. `strace -T` shows each ioctl taking ~1ms. | Set `spi.max_speed_hz = 10_000_000` explicitly. Validate with `spidev_test -s 10000000`. |
| Publish rate stuck at 83–87Hz | `xfer2()` called inside ROS2 timer callback — executor stalled | `ros2 topic hz /imu/raw` shows ~87Hz. `strace -T` shows ioctl calls ~55µs each but callback period is >11ms. | Move `xfer2()` to a dedicated Python thread. Push frames via `queue.Queue`. Timer callback only dequeues and publishes. |
| No messages received on a topic despite publisher running | QoS mismatch — publisher BEST_EFFORT, subscriber RELIABLE (or vice versa) | `ros2 topic hz /imu/raw` shows 0Hz at subscriber. `ros2 topic info -v /imu/raw` shows mismatched Reliability fields between publisher and subscriber. | Both publisher and subscriber must use compatible QoS. Use `qos_profile_sensor_data` (BEST_EFFORT) on both ends for sensor topics. |
| rviz2 shows no data / "No tf data" | Wrong `frame_id` in message header | Check `ros2 topic echo /imu/raw | grep frame_id`. If it says `"imu"` but your TF tree has `"imu_link"`, TF2 can't look up the transform. | Set `msg.header.frame_id = 'imu_link'` (match exactly what's in your static TF broadcaster). |
| EKF diverges / orientation drifts without physical motion | Timestamp captured too late (stale by 1–3ms) | `ros2 topic delay /imu/raw` shows >2ms end-to-end latency. The EKF integrates angular velocity using dt that includes the latency. Symptoms appear as slow orientation drift. | Capture `CLOCK_MONOTONIC_RAW` **immediately** after `xfer2()` returns, before any decode or message construction. Pass the nanosecond value through the queue to the publisher. |
| EKF diverges immediately to NaN | Covariance matrix has zeros on diagonal | Check your Imu message: `msg.linear_acceleration_covariance` all zeros. A zero covariance diagonal means "infinite precision" → Kalman gain computation divides by zero → NaN propagates. | Set `msg.linear_acceleration_covariance[0] = -1.0` (unknown) OR fill the full 3×3 diagonal with your calibrated variance values (e.g., `0.01`). Never leave as all-zeros. |
| Intermittent 2ms latency spikes in SPI timing despite PREEMPT-RT | `irqbalance` migrating the SPI IRQ to a different CPU | Spikes appear every few hundred seconds, disappear on node restart. `cat /proc/interrupts | grep spi` shows the SPI IRQ moving between CPUs over time. | `sudo systemctl disable irqbalance && sudo systemctl stop irqbalance`. Pin the SPI IRQ manually: `echo 8 > /proc/irq/<N>/smp_affinity` (CPU3). |
| `rosbag2 record` degrades IMU publish rate | SQLite WAL flush: default storage backend blocks all DDS writes for ~40ms every ~4 seconds | Start recording, watch `ros2 topic hz /imu/raw` drop from 100Hz to periodic stuttering. `strace` on the ros2 bag process shows `fsync()` calls taking 20–40ms. | Use MCAP storage: `ros2 bag record --storage mcap /imu/raw`. Or set BEST_EFFORT QoS on all sensor topics so DDS backpressure doesn't stall the publisher. |
| `spidev_test` works but Python script gets garbage | Wrong SPI mode (CPOL/CPHA mismatch) | `spidev_test` uses mode 0 by default. Your Python script might have `spi.mode = 1` or `spi.mode = 2`. Logic analyzer: check CLK idle state and sample edge. | Match `spi.mode` exactly to your STM32 configuration. STM32 `SPI_CPOL_Low + SPI_CPHA_1Edge` = mode 1, not mode 0. |
| `colcon build` succeeds but `ros2 run spi_bridge spi_bridge` says "executable not found" | `source install/setup.bash` not run after build | The new package is not yet on the ROS2 package path. | Run `source ~/ros2_ws/install/setup.bash` after every `colcon build`. Add this to `~/.bashrc` for convenience. |
| EKF ignores IMU entirely — only uses odometry | IMU topic not listed in `robot_localization` YAML, or `imu0_config` has all `false` | `robot_localization` debug output shows "imu0: 0 measurements fused." Topic name typo in YAML, or all fuse flags are `false`. | Verify `imu0: /imu/raw` matches the actual topic name. Ensure at least `vroll, vpitch, vyaw` and `ax, ay, az` are `true` in `imu0_config`. |
| `ros2 topic delay /imu/raw` shows 50ms | rosbag2 recording on same machine + RELIABLE QoS causing backpressure | Delay grows under recording load. Subscriber queue fills up. | Switch to BEST_EFFORT QoS everywhere for sensor topics, and use MCAP storage for rosbag2. |
| Thermal throttling looks like SPI timeouts | Jetson CPU frequency drops 30% under sustained load | `tegrastats` shows CPU freq below max. `sudo nvpmodel -q` shows non-MAXN mode. `cyclictest` Max latency increases over time. | `sudo nvpmodel -m 0 && sudo jetson_clocks` before any timing test. Ensure adequate cooling on heatsink. |

---

## Quick Reference Card

```
# ── spidev hardware validation ──────────────────────────────────────────────
ls /dev/spidev*                              # verify /dev/spidev0.0 exists
sudo spidev_test -D /dev/spidev0.0 \         # hardware round-trip test
    -s 10000000 -p "AABBCCDD" -v

# ── Jetson RT setup (run before any SPI node) ─────────────────────────────
sudo nvpmodel -m 0 && sudo jetson_clocks    # max CPU/GPU clocks, disable throttling
sudo systemctl disable irqbalance           # stop IRQ migration
sudo systemctl stop irqbalance              # stop immediately
# Edit /boot/extlinux/extlinux.conf: add isolcpus=3 nohz_full=3 rcu_nocbs=3
echo 8 > /proc/irq/$(cat /proc/interrupts | grep spi | awk '{print $1}' | tr -d ':')/smp_affinity

# ── RT latency measurement ──────────────────────────────────────────────────
sudo cyclictest -m -sp99 -t4 -l100000       # 100k samples, 4 threads; Max should be <100µs
sudo strace -T -e trace=ioctl python3 spi_reader.py 2>&1 | grep SPI  # real ioctl duration

# ── Thread RT setup (in Python) ─────────────────────────────────────────────
# From shell: sudo chrt -f 90 taskset -c 3 python3 spi_reader.py
import ctypes; libc = ctypes.CDLL('libc.so.6', use_errno=True)
class SP(ctypes.Structure): _fields_=[('sched_priority',ctypes.c_int)]
libc.sched_setscheduler(0, 1, ctypes.byref(SP(90)))   # SCHED_FIFO priority 90
import os; os.sched_setaffinity(0, {3})               # pin to CPU3

# ── ROS2 health checks ───────────────────────────────────────────────────────
ros2 topic hz /imu/raw                       # should show ~100.0Hz
ros2 topic delay /imu/raw                    # should show <5ms
ros2 topic info -v /imu/raw                  # shows QoS for all connections (detect mismatch)
ros2 run tf2_tools view_frames               # verify TF tree; evince frames.pdf to view
ros2 run tf2_ros tf2_echo base_link imu_link # verify static transform exists

# ── QoS — always use for sensor topics ─────────────────────────────────────
from rclpy.qos import qos_profile_sensor_data
pub = self.create_publisher(Imu, '/imu/raw', qos_profile_sensor_data)

# ── Covariance — never leave as zero ────────────────────────────────────────
msg.linear_acceleration_covariance[0] = -1.0   # -1.0 = unknown (lets EKF use YAML config)
msg.angular_velocity_covariance[0]    = -1.0
msg.orientation_covariance[0]         = -1.0

# ── Timestamp — capture immediately after xfer2() ──────────────────────────
import ctypes; librt = ctypes.CDLL('librt.so.1')
class TS(ctypes.Structure): _fields_=[('tv_sec',ctypes.c_long),('tv_nsec',ctypes.c_long)]
def now_ns():
    t=TS(); librt.clock_gettime(4, ctypes.byref(t)); return t.tv_sec*10**9+t.tv_nsec
rx = spi.xfer2([0]*64); ts_ns = now_ns()  # capture BEFORE ANY DECODE

# ── robot_localization launch ────────────────────────────────────────────────
ros2 launch robot_localization ekf.launch.py   # verify before ROS2 node
# In ekf_config.yaml: imu0: /imu/raw, imu0_config: vroll,vpitch,vyaw,ax,ay,az = true
# NEVER put all zeros in covariance matrices — causes NaN immediately

# ── rosbag2 for sensors — use MCAP storage ──────────────────────────────────
ros2 bag record --storage mcap /imu/raw /wheel_speed /odom   # avoids SQLite WAL stalls
```
