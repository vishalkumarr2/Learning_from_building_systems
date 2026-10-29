# 09 — Two-Layer Architecture: MCU ↔ Jetson
### The SPI bus between real-time and best-effort — where timing contracts live

**Prerequisite:** `07-cascaded-loops.md` (MCU cascade), `08-trajectory-tracking.md` (Jetson path following)
**Unlocks:** `10-advanced-control.md` (when this architecture isn't enough)

---

## Why Should I Care? (Practical Context)

The warehouse robot runs **two computers** that must cooperate in real-time:

| | STM32 MCU | Jetson (Linux + ROS2) |
|---|---|---|
| **Clock** | 168 MHz bare-metal / RTOS | 1.5 GHz Linux + preempt_RT |
| **Latency** | Deterministic (µs) | Best-effort (ms) |
| **Role** | Motor control, safety | Navigation, planning, perception |
| **Rate** | 10–20 kHz ISRs | 20–50 Hz ROS nodes |
| **Failure mode** | Watchdog reset | Process crash, OOM kill |

**The SPI bus is the handshake** between these two worlds. When the handshake breaks, the robot drifts, stops, or (worst case) drives into a rack.

**Real AMR failures from architecture issues:**
- Jetson GC pause (50 ms) → no SPI update → MCU watchdog → emergency stop (Incident-C)
- SPI frame corruption → MCU interprets noise as speed command → motor spike (Incident-A)
- Clock drift between MCU and Jetson → cmd_vel timestamps misaligned → estimator rejects data

---

# PART 1 — THE COMMUNICATION PROTOCOL

## 1.1 SPI Bus Overview

```
Jetson (SPI Master)              STM32 (SPI Slave)
┌──────────────┐                 ┌──────────────┐
│              │────── SCLK ────→│              │
│              │────── MOSI ────→│    SPI DMA   │
│              │←───── MISO ─────│              │
│              │────── CS   ────→│              │
│   SPI @      │                 │   SPI @      │
│   2 MHz      │                 │   2 MHz      │
└──────────────┘                 └──────────────┘
```

**AMR SPI parameters:**
- Clock: 2 MHz (500 ns per bit)
- Frame size: 64 bytes (32 bytes each direction)
- Transfer rate: 1 kHz (every 1 ms)
- Protocol: nanopb (protocol buffers for embedded)

## 1.2 Message Format

**Jetson → MCU (command frame):**

```protobuf
message MotorCommand {
    uint32 timestamp_us = 1;    // Jetson's microsecond clock
    float left_speed_cmd = 2;   // rad/s, at wheel
    float right_speed_cmd = 3;  // rad/s, at wheel
    uint8 control_mode = 4;     // 0=idle, 1=speed, 2=position, 3=raw_duty
    uint8 enable = 5;           // 0=disabled, 1=enabled
    uint16 crc16 = 6;           // CRC-16/CCITT
}
```

**MCU → Jetson (feedback frame):**

```protobuf
message MotorFeedback {
    uint32 timestamp_us = 1;     // MCU's microsecond clock
    float left_speed = 2;        // rad/s, measured
    float right_speed = 3;       // rad/s, measured
    float left_current = 4;      // Amps, measured
    float right_current = 5;     // Amps, measured
    int32 left_encoder = 6;      // Cumulative encoder count
    int32 right_encoder = 7;     // Cumulative encoder count
    uint8 fault_flags = 8;       // Overcurrent, overtemp, encoder error
    uint16 crc16 = 9;
}
```

## 1.3 CRC Validation

Every frame includes a CRC-16 checksum. If the CRC doesn't match, the MCU **ignores the command** and holds the last valid command (for up to the watchdog timeout).

```c
// CRC-16/CCITT (polynomial 0x1021)
uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000)
                crc = (crc << 1) ^ 0x1021;
            else
                crc <<= 1;
        }
    }
    return crc;
}
```

**AMR failure rate:** Approximately 1 in 10,000 SPI frames is corrupted (0.01%), usually from EMI near the motor drivers. The CRC catches all of these.

---

# PART 2 — TIMING CONTRACTS

## 2.1 The Timing Budget

```
                     1 ms SPI cycle
        ├──────────────────────────────────┤
        
Jetson: │ Prepare cmd ─→ SPI write ─→ Wait │
        │     100 µs       200 µs     700 µs│
        
MCU:    │ SPI DMA recv ─→ Validate ─→ Apply │
        │     200 µs        10 µs      10 µs │
        │                              ↓      │
        │           Speed PID runs at 10 kHz  │
        │           (every 100 µs within)     │
```

**Key contract:** The Jetson must deliver a new command **every 1 ms ± 0.2 ms**. If a command doesn't arrive within 2 ms (2 missed cycles), the MCU enters **hold mode** (maintains last speed command). If no command arrives within 10 ms (10 missed cycles), the MCU enters **emergency deceleration**.

## 2.2 Watchdog Architecture

```
                    Timer-based watchdog
                    ┌─────────────────┐
                    │  Counter = 10   │
                    │  Decrement @    │
                    │  1 kHz          │
                    │                 │
  SPI valid frame ──→  Reset to 10   │
                    │                 │
  Counter = 5 ─────→  HOLD mode      │ (reduce speed by 50%)
  Counter = 0 ─────→  E-STOP         │ (brake motors)
                    └─────────────────┘
```

```c
// MCU watchdog for SPI communication
#define WDG_RELOAD     10    // 10 ms timeout
#define WDG_HOLD_LEVEL  5    // 5 ms → enter hold

volatile uint8_t spi_watchdog_counter = 0;

// Called in SPI RX complete callback
void spi_frame_received_ok(void) {
    spi_watchdog_counter = WDG_RELOAD;
}

// Called at 1 kHz by timer
void watchdog_tick(void) {
    if (spi_watchdog_counter > 0) {
        spi_watchdog_counter--;
    }
    
    if (spi_watchdog_counter == 0) {
        // E-STOP: brake motors immediately
        motor_brake();
        set_fault_flag(FAULT_COMM_LOST);
    } else if (spi_watchdog_counter <= WDG_HOLD_LEVEL) {
        // HOLD: reduce speed command by 50%
        speed_setpoint_left /= 2;
        speed_setpoint_right /= 2;
    }
}
```

## 2.3 Jetson-Side Timing

The ROS2 node that sends SPI commands must respect the timing contract:

```python
import rclpy
from rclpy.node import Node
import spidev
import time

class MotorBridge(Node):
    def __init__(self):
        super().__init__('motor_bridge')
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 2000000
        
        # Timer at 1 kHz (1 ms period)
        self.timer = self.create_timer(0.001, self.spi_callback)
        
        self.last_cmd_vel = (0.0, 0.0)  # (v, omega)
        self.cmd_vel_stamp = self.get_clock().now()
        
    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = (msg.linear.x, msg.angular.z)
        self.cmd_vel_stamp = self.get_clock().now()
    
    def spi_callback(self):
        # Check cmd_vel freshness
        age = (self.get_clock().now() - self.cmd_vel_stamp).nanoseconds / 1e9
        if age > 0.2:  # 200 ms stale → stop
            v, omega = 0.0, 0.0
        else:
            v, omega = self.last_cmd_vel
        
        # Inverse kinematics
        wL = (v - omega * WHEELBASE / 2) / WHEEL_RADIUS
        wR = (v + omega * WHEELBASE / 2) / WHEEL_RADIUS
        
        # Build command frame
        frame = build_motor_command(wL, wR)
        
        # SPI transfer
        response = self.spi.xfer2(frame)
        
        # Parse feedback
        feedback = parse_motor_feedback(response)
        self.publish_odom(feedback)
```

---

# PART 3 — LATENCY ANALYSIS

## 3.1 End-to-End Latency

From Nav2 decision to motor torque change:

| Stage | Latency | Source |
|-------|---------|--------|
| Nav2 controller → cmd_vel publish | 0–20 ms | ROS2 DDS, timer jitter |
| cmd_vel → motor_bridge callback | 0–5 ms | ROS2 intra-process: ~0.1 ms, inter-process: ~5 ms |
| Motor bridge processing | 0.1 ms | Inverse kinematics |
| SPI transfer | 0.2 ms | 64 bytes @ 2 MHz |
| MCU validates + applies | 0.01 ms | CRC check + register write |
| Next speed PID cycle | 0–0.1 ms | Waits for next 10 kHz tick |
| Current PID → PWM update | 0.05 ms | One current loop cycle |
| Motor electrical response | 0.5 ms | Electrical time constant |
| **Total (typical)** | **~5 ms** | |
| **Total (worst case)** | **~30 ms** | |

## 3.2 Why Latency Matters

A 30 ms delay at 0.5 m/s means the robot travels **15 mm** before responding to a command change. For corridor driving, this is fine. For docking (±5 mm precision), this is borderline.

**Control theory perspective:** A time delay $\tau$ in the loop adds phase lag:

$$\angle G(j\omega) = -\omega \tau \text{ radians}$$

At the speed loop's 50 Hz bandwidth ($\omega = 100\pi$):
$-100\pi \times 0.030 = -9.4°$ of extra phase lag.

The phase margin typically has ~40° to spare, so 9.4° is acceptable. But at 100 Hz bandwidth, the lag would be 18.8° — getting tight.

## 3.3 The cmd_vel Gap Problem

**robot-specific:** If the Jetson's Nav2 controller runs at 20 Hz and the motor bridge runs at 1 kHz, there are ~50 SPI cycles between each new cmd_vel. The motor bridge sends the **same** cmd_vel 50 times.

If Nav2 misses a cycle (garbage collection, CPU spike), there's a **100 ms gap** in cmd_vel updates. The velocity smoother helps, but the MCU sees stale commands.

**Worse:** If the Nav2 controller's timer fires late, the cmd_vel can be timestamped incorrectly, causing the estimator to compute wrong odometry deltas.

---

# PART 4 — CLOCK SYNCHRONIZATION

## 4.1 The Problem

The Jetson's clock and the MCU's clock are independent. Over time, they drift:
- Jetson: NTP-synced, typical drift ~1 ms/hour
- MCU: Crystal oscillator, drift ~50 ppm = 50 µs/second = 180 ms/hour

After 1 hour, the clocks can be 200 ms apart. If the Jetson stamps a command at time $t$ but the MCU thinks the current time is $t + 200ms$, the estimator's time-interpolation breaks.

## 4.2 Solution: Timestamp Exchange

Every SPI frame includes timestamps from both sides. The Jetson computes the offset:

```python
class ClockSync:
    def __init__(self):
        self.offset_us = 0
        self.alpha = 0.01  # Low-pass filter coefficient
    
    def update(self, jetson_send_time_us, mcu_timestamp_us, jetson_recv_time_us):
        """Estimate MCU clock offset using SPI round-trip."""
        # Round-trip time
        rtt = jetson_recv_time_us - jetson_send_time_us
        
        # Assume symmetric delay: MCU received at send_time + rtt/2
        estimated_mcu_time_at_midpoint = jetson_send_time_us + rtt // 2
        
        # Offset = Jetson time - MCU time
        raw_offset = estimated_mcu_time_at_midpoint - mcu_timestamp_us
        
        # Exponential moving average to filter noise
        self.offset_us = int(self.alpha * raw_offset + (1 - self.alpha) * self.offset_us)
    
    def mcu_to_jetson_time(self, mcu_time_us):
        return mcu_time_us + self.offset_us
```

---

# PART 5 — FAILURE MODES AND SAFETY

## 5.1 Failure Mode Table

| Failure | Detection | MCU response | Jetson response |
|---------|-----------|-------------|----------------|
| SPI frame corrupt | CRC mismatch | Hold last valid command | Log warning, retry |
| SPI completely lost | Watchdog timeout (10 ms) | Emergency brake | Publish `/motor_fault` |
| Jetson frozen/crashed | No SPI writes | Watchdog → brake | (dead) Systemd restarts |
| MCU frozen | No SPI response | (dead) Hardware WDG resets | Detect stale feedback → stop |
| Motor overcurrent | Shunt ADC > threshold | Immediate PWM disable | Publish fault diagnostic |
| Encoder failure | Speed estimate unreasonable | Reduce to open-loop + brake | Log, request service |

## 5.2 The "Never Trust the Other Side" Principle

**MCU rule:** Never trust that Jetson commands are sane. Always validate:
- Speed command within ±$v_{max}$? If not, clamp.
- CRC valid? If not, use last good command.
- Command fresh? If not, decelerate.

**Jetson rule:** Never trust that MCU feedback is accurate. Always validate:
- Speed physically possible? (> 2× max speed = encoder error)
- Current within range? (negative current = sensor failure)
- Encoder monotonically consistent with speed sign?

```c
// MCU-side command validation
bool validate_motor_command(MotorCommand *cmd) {
    // Range check
    if (fabsf(cmd->left_speed_cmd) > MAX_WHEEL_SPEED) return false;
    if (fabsf(cmd->right_speed_cmd) > MAX_WHEEL_SPEED) return false;
    
    // Mode check
    if (cmd->control_mode > 3) return false;
    
    // Rate-of-change check (no impossible acceleration)
    float dv_left = fabsf(cmd->left_speed_cmd - prev_cmd.left_speed_cmd);
    if (dv_left > MAX_ACCEL * SPI_PERIOD) return false;
    
    return true;
}
```

## 5.3 Graceful Degradation

When things go wrong, degrade smoothly:

```
NORMAL → REDUCED → HOLD → STOP → BRAKE
  ↑                                 │
  └─────── Recovery ────────────────┘
  
NORMAL:  Full speed, full control
REDUCED: 50% max speed, increased monitoring
HOLD:    Maintain last velocity, start decelerating
STOP:    Zero velocity command, PID still active
BRAKE:   PWM off, motor shorted (dynamic braking), no PID
```

---

## Checkpoint Questions

1. Why use SPI instead of UART for MCU↔Jetson communication?
2. What happens if the SPI CRC fails? Why not just accept the frame anyway?
3. The MCU watchdog triggers after 10 ms of no SPI. Is 10 ms too fast, too slow, or about right? Why?
4. End-to-end latency from Nav2 cmd_vel to motor torque is ~5 ms typical, ~30 ms worst case. What physical distance does the robot travel in the worst case at 0.5 m/s?
5. The Jetson's garbage collector pauses for 80 ms. Trace the sequence of events on the MCU side.
6. MCU clock drifts 50 ppm from Jetson. After 10 minutes, what's the offset in milliseconds?
7. Why must the MCU validate command rate-of-change, not just absolute range?

---

## Key Takeaways

- **Two-layer architecture:** MCU (deterministic, µs) + Jetson (flexible, ms) connected via SPI
- **nanopb protocol** with CRC-16 ensures data integrity across the noisy power bus
- **Timing contracts** (1 ms ± 0.2 ms SPI cycle, 10 ms watchdog) are the foundation of reliability
- **End-to-end latency** (~5–30 ms) determines the achievable control bandwidth from the Jetson layer
- **Clock synchronization** via timestamp exchange prevents estimator errors
- **"Never trust the other side"** — both MCU and Jetson validate all incoming data
- **Graceful degradation:** Normal → Reduced → Hold → Stop → Brake
