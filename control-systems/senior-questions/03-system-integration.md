# Senior Interview Questions — System Integration

**Purpose:** Questions about the MCU↔Jetson boundary, real-world failure modes, and architectural decisions.
**Level:** Senior systems/robotics engineer (5+ years)

---

## Question 1: Latency Budget Allocation

> "You have a 50 ms budget from Nav2 decision to wheel motion. How do you allocate it?"

**Expected weak answer:** "Just make everything fast."

**Deep answer:** Work backward from the physics:

```
Budget: 50 ms (20 Hz control loop)

Nav2 path computation:     0-15 ms  (variable, depends on costmap)
ROS2 DDS serialization:    1-3 ms   (topic publish + subscribe)
Velocity smoother:         1-2 ms   (acceleration limiting)
SPI to MCU:                0.5-1 ms (transfer + CRC check)
MCU speed PID:             0.01 ms  (trivial computation)
Motor electrical response: 1-2 ms   (current rise in L/R)
Motor mechanical response: 5-40 ms  (J/B time constant)
                           ─────────
Total:                     8.5-63 ms (worst case exceeds budget!)
```

**Key insight:** The budget is not a hard deadline — it's a **jitter sensitivity analysis.**

- The motor mechanical response (5-40 ms) is the dominant term and is **fixed** by physics
- The controllable latency is Nav2 + DDS + SPI ≈ 2-19 ms
- The **variation** matters more than the **absolute** latency. A constant 40 ms delay is fine (the controller adapts). A latency that varies 10-40 ms causes oscillation.
- Allocate: make the controllable parts **consistent**, not just fast. CPU pinning and RT priority reduce jitter more than raw speed.

---

## Question 2: SPI Failure Modes

> "The SPI link between Jetson and MCU drops packets. How do you handle this?"

**Expected weak answer:** "Retransmit lost packets."

**Deep answer:** **Never retransmit in real-time control.** A retransmitted packet arrives late — the data is stale.

**SPI failure taxonomy:**
1. **Corruption** (CRC fail): Data arrived but is wrong
2. **Dropout** (no response): MCU didn't respond within timeout
3. **Stuck bus** (SCK frozen): Hardware failure, bus locked
4. **Desync** (byte offset): Master and slave are out of frame alignment

**Handling strategy per type:**

```
Corruption: 
  → Discard this frame
  → Use previous good data (one frame old)
  → Increment error counter
  → If 3 consecutive CRC fails → re-sync (toggle CS, send sync pattern)

Dropout:
  → Use previous good data
  → Start watchdog timer
  → If > 10 ms of dropouts → enter reduced mode
  → If > 50 ms → active braking

Stuck bus:
  → Toggle GPIO to reset SPI peripheral
  → Re-initialize SPI
  → If still stuck → switch to UART fallback (if available)

Desync:
  → Send magic byte sequence (0xAA, 0x55) to re-frame
  → Wait for sync acknowledgment
  → Resume normal framing
```

**Critical principle:** The MCU must **never assume the Jetson is healthy.** Every received frame must be validated independently. The MCU operates autonomously if communication is lost.

---

## Question 3: Clock Synchronization

> "The Jetson timestamps a cmd_vel at T=1.000s. The MCU receives it and its local clock reads T=1.003s. What's the real latency — 3 ms or unknown?"

**Expected weak answer:** "3 ms."

**Deep answer:** **Unknown.** The clocks are not synchronized.

- The Jetson has a crystal oscillator accurate to ±20 ppm
- The MCU has an internal RC oscillator accurate to ±1%
- After 1 second, the MCU clock has drifted by up to 10 ms from the Jetson clock
- The 3 ms difference could be: 3 ms real latency, or 0 ms latency with 3 ms clock drift, or 13 ms latency with -10 ms clock drift

**Solutions:**

1. **Periodic sync pulse:** Jetson sends a GPIO pulse every 100 ms. MCU measures its own timer at the pulse edge. Calculate drift: $\Delta_{drift} = (T_{mcu}[n] - T_{mcu}[n-1]) - 100.0$ ms. Compensate timestamps.

2. **Round-trip measurement:** Jetson sends timestamp $T_1$, MCU echoes with its own $T_2$, Jetson receives at $T_3$. One-way latency ≈ $(T_3 - T_1) / 2$. No clock sync needed, but assumes symmetric latency.

3. **Crystal oscillator on MCU:** Replace RC with 8 MHz crystal. Drift drops to ±20 ppm = 0.02 ms per second. Acceptable for most control applications.

**OKS approach:** Option 3 (crystal on MCU) + option 1 (periodic sync) as validation. The sync pulse also serves as a heartbeat — if the MCU doesn't see it, the Jetson is in trouble.

---

## Question 4: The Shared State Problem

> "Nav2 publishes cmd_vel. The velocity smoother rate-limits it. The motor bridge converts it to wheel speeds. The MCU executes it. Three nodes, one intent. What can go wrong?"

**Expected weak answer:** "Nothing if DDS QoS is configured correctly."

**Deep answer:** **The distributed state consistency problem.** Multiple nodes each have a different view of the robot's state.

**Failure mode 1: Temporal inconsistency**
- Nav2 computed cmd_vel using TF transform from 50 ms ago
- The smoother applies acceleration limits based on the velocity it last saw (which might be 20 ms old)
- The motor bridge converts using its cached wheel separation (which doesn't account for tire wear)
- Each node is correct locally but the chain is wrong globally

**Failure mode 2: Race condition on mode switch**
- Nav2 publishes the last cmd_vel for the current goal
- Simultaneously, the behavior tree cancels the goal
- The smoother receives the cmd_vel but not the cancel
- The motor bridge accelerates while Nav2 thinks it's stopped

**Failure mode 3: QoS mismatch**
- Nav2 uses RELIABLE QoS (guaranteed delivery)
- Motor bridge uses BEST_EFFORT (low latency)
- A bridge node converts between them
- Under load, the bridge drops packets → motor bridge gets intermittent cmd_vel

**Architectural fix:** **Single source of truth with monotonic timestamps.**

```python
class CmdVelFrame:
    stamp: Time          # When was this computed
    sequence: uint32     # Monotonically increasing
    v: float64           # Linear velocity
    omega: float64       # Angular velocity
    valid_until: Time    # Expiry timestamp
    source: string       # Who generated this
```

Every downstream consumer checks `sequence` (detect missing frames) and `valid_until` (reject stale commands).

---

## Question 5: Graceful Degradation

> "Design the degradation hierarchy when things start failing."

**Expected weak answer:** A simple table of "if X fails, do Y."

**Deep answer:** Multi-dimensional degradation with independent axes:

```
Communication axis:
  FULL      → All SPI + DDS + TF working
  DEGRADED  → SPI okay, DDS intermittent (use last good TF)
  MINIMAL   → SPI only, no higher-level commands
  ISOLATED  → SPI lost, MCU autonomous
  
Sensing axis:
  FULL      → LiDAR + cameras + encoders + IMU
  REDUCED   → LiDAR down → reduce speed, wider margins
  MINIMAL   → Encoders only → dead reckoning, 0.1 m/s max
  BLIND     → Encoders failed → immediate stop

Power axis:
  FULL      → Battery > 30%
  LOW       → Battery 10-30% → reduce max speed
  CRITICAL  → Battery < 10% → navigate to charger only
  DEAD      → Battery < 5% → stop and call for help

Control axis:
  FULL      → Cascade PID + feedforward + gain scheduling
  REDUCED   → PID only, no feedforward (higher error, still safe)
  BASIC     → P-only control (jerky but functional)
  OPEN      → Fixed PWM (emergency crawl)
```

**Key design principle:** Each axis degrades independently. A robot with DEGRADED communication + REDUCED sensing + FULL power + FULL control can still operate (at reduced speed with wider margins). The system only stops when **any axis hits its terminal state**.

**Implementation:** Finite state machine with hysteresis (don't flap between states):

```c
typedef struct {
    DegradationLevel comm;
    DegradationLevel sensing;
    DegradationLevel power;
    DegradationLevel control;
} SystemHealth;

OperatingMode compute_mode(SystemHealth h) {
    if (h.comm == ISOLATED || h.sensing == BLIND || 
        h.power == DEAD || h.control == OPEN)
        return EMERGENCY_STOP;
    
    // Take the worst axis
    int worst = max(h.comm, h.sensing, h.power, h.control);
    
    switch (worst) {
        case FULL:     return NORMAL;
        case DEGRADED: return REDUCED_SPEED;
        case REDUCED:  return SLOW_CRAWL;
        case MINIMAL:  return STOP_AND_HOLD;
    }
}
```

---

## Question 6: Firmware Update While Running

> "Can you update the motor controller firmware without stopping the fleet?"

**Expected weak answer:** "Just OTA update and reboot."

**Deep answer:** The motor controller is **safety-critical.** Firmware update requires a careful protocol:

1. **Dual-bank flash:** MCU has two firmware banks (A and B). Currently running from A.
2. **Download to B** while A is still running. Robot continues operating normally.
3. **Validate B:** Check CRC, verify it's a valid image, check version compatibility with Jetson.
4. **Schedule switchover:** Wait for the robot to be stationary, at a charging station, with no pending tasks.
5. **Atomic switch:** Mark B as active, reboot. First thing B does: self-test (motor drives, sensors, SPI communication).
6. **Rollback:** If self-test fails, switch back to A automatically.
7. **Confirm:** Robot operates on B for 5 minutes. If no anomalies, mark A as "previous good" (not delete it).

**Never update while moving.** The reboot takes 200-500 ms. At 0.5 m/s, the robot travels 10-25 cm uncontrolled. In a warehouse with 1.2 m aisles, that's unacceptable.

**Never update all robots simultaneously.** Rolling update: 10% at a time, 30-minute observation between batches.

---

## Question 7: When PID Isn't Enough

> "Give me three real scenarios on an OKS robot where PID fails and you need something else."

**Expected weak answer:** Generic examples from textbooks.

**Deep answer — OKS-specific:**

**Scenario 1: Loaded vs unloaded robot**
- Empty robot: $J = 0.5$ kg·m². Loaded: $J = 5.0$ kg·m² (10× heavier).
- PID tuned for empty overshoots violently when loaded.
- Solution: **Gain scheduling** on payload weight (load cell + encoder current draw).

**Scenario 2: Tight 90° corner at speed**
- The robot needs to decelerate, turn, and accelerate — three phases
- PID reacts to error. At the turn apex, error is zero (you're at the waypoint) but you need to be **accelerating out of the turn**
- Solution: **Feedforward** from the trajectory planner. The FF term provides the turn deceleration and acceleration. PID only corrects residual errors.

**Scenario 3: Floor transition (concrete → epoxy → metal plate)**
- Friction coefficient changes suddenly. Wheels slip on metal plates.
- PID integral winds up during slip (actual speed drops, error grows, integral accumulates)
- When friction returns, the wound-up integral causes a lurch
- Solution: **Slip detection** (compare encoder speed to IMU acceleration) + **integral reset** on slip detection + **disturbance observer** to estimate and compensate the friction change.

---

## Question 8: SPI vs CAN vs UART

> "Why does OKS use SPI for Jetson↔MCU? Why not CAN or UART?"

**Expected weak answer:** "SPI is faster."

**Deep answer:** It's a tradeoff matrix:

| Feature | SPI | CAN | UART |
|---------|-----|-----|------|
| Speed | 10+ MHz | 1 Mbps | 1-3 Mbps |
| Distance | < 20 cm (PCB) | 40 m (bus) | 15 m (RS-485) |
| Wires | 4 (MOSI/MISO/SCK/CS) | 2 (CANH/CANL) | 2 (TX/RX) |
| Latency | < 50 µs per frame | 0.1-1 ms (arbitration) | 0.1 ms |
| Error detection | None built-in | CRC-15 built-in | None built-in |
| Multi-device | CS per device | Bus (128 nodes) | Point-to-point |
| Full duplex | Yes | No | Yes |
| CPU overhead | Medium (DMA helps) | Low (hardware CRC) | Medium |

**Why SPI for OKS:**
- Jetson and MCU are on the same PCB → short distance (SPI weakness irrelevant)
- Need **bidirectional data** every 1 ms: command down + status up (full duplex advantage)
- Need **low latency** for tight control loops (SPI < 50 µs vs CAN 0.1-1 ms)
- Frame size is 32-64 bytes → fits in one SPI transaction

**When CAN is better:**
- Multiple motor controllers (one per wheel) → CAN bus is natural
- Long cable runs (chassis wiring)
- Need guaranteed message delivery with priority arbitration

**When UART is better:**
- Debugging and logging (printf over UART)
- GPS or sensor modules that speak UART natively
- Simplest possible connection (2 wires)

---

## Question 9: The Timestamp Paradox

> "A ROS2 message has `header.stamp`. Should the MCU trust this timestamp?"

**Expected weak answer:** "Yes, it's the authoritative time."

**Deep answer:** **Never trust the other side's timestamp for control decisions.**

Problems:
1. **Clock drift:** Jetson and MCU clocks are not synchronized (see Question 3)
2. **Stale data:** The message might have been sitting in a DDS queue for 20 ms. The stamp says when it was **created**, not when the MCU **received** it.
3. **Spoofing:** In a safety context, trusting external timestamps means a Jetson bug could make the MCU think data is fresh when it's minutes old.

**Correct usage:**

```c
// MCU receives a SPI frame with Jetson timestamp
uint32_t jetson_stamp = frame.timestamp;  // Informational only
uint32_t mcu_rx_time = get_local_time();  // Authority for freshness

// Freshness check uses MCU clock
if (mcu_rx_time - last_good_rx > STALENESS_THRESHOLD) {
    enter_degraded_mode();
}

// Jetson timestamp is used for:
// - Logging and post-mortem analysis
// - Estimating communication latency (with clock sync correction)
// - Detecting Jetson time jumps (NTP corrections, sim time glitches)
```

**OKS rule: The MCU clock is king for safety decisions.** The Jetson timestamp is informational metadata.

---

## Question 10: Design Review Question

> "Your teammate proposes: 'Let's run the PID on the Jetson instead of the MCU. The Jetson has more compute power and we can use floating point.' Argue against this."

**Expected weak answer:** "The MCU is more real-time."

**Deep answer — structured argument:**

**1. Latency chain gets longer:**
- On MCU: encoder → PID → PWM. Total: < 10 µs. All in one ISR.
- On Jetson: encoder → SPI to Jetson → ROS2 DDS → PID node → DDS → SPI to MCU → PWM. Total: 5-30 ms. 500-3000× slower.

**2. Jitter kills control performance:**
- Jetson runs Linux (not real-time). Timer callbacks have 1-50 ms jitter under load.
- Current loop at 10 kHz with 5 ms jitter = unstable. Not viable.
- Even speed loop at 1 kHz with 5 ms jitter = audible noise, poor tracking.

**3. Single point of failure:**
- If Jetson crashes, all motor control is lost. Robot coasts uncontrolled.
- With PID on MCU: Jetson crash → MCU detects loss → MCU brakes autonomously.
- The MCU is the **safety layer.** It must function independently.

**4. The Jetson IS more powerful — use it for what it's good at:**
- Path planning (search algorithms, costmaps) — Jetson
- Localization (particle filters, AMCL) — Jetson
- Trajectory generation (optimization, MPC) — Jetson
- Inner-loop motor control (PID at kHz rates) — MCU
- Safety monitoring (watchdog, braking) — MCU

**5. Cost of "more compute":**
- Floating-point PID is slightly easier to write but wastes Jetson CPU on trivial math
- That CPU is better spent on perception (cameras, LiDAR processing)
- The MCU costs $3. The Jetson costs $300. Use each where it adds value.

**The two-layer split exists for a reason.** It's not a legacy constraint — it's a deliberate safety and performance architecture.
