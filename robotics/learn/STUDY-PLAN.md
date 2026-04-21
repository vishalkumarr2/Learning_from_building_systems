# Unified Study Plan: Electronics → Protocols → Embedded Systems
### From resistors to a 100Hz STM32 ↔ Jetson SPI bridge
### Total: ~120–130 hours across 8 weeks

---

## How This Plan Works

```
Week 1:  Electronics Essentials       ──→  You understand the PHYSICAL layer
Week 2:  SPI-Focused Protocol Deep Dive ──→  You understand the WIRE layer
Week 3:  Zephyr Foundations            ──→  You can BUILD firmware
Week 4:  Sensors + ZBus + nanopb       ──→  You can READ sensors + ENCODE data
Week 5–6: SPI Slave DMA               ──→  You can DMA at 100Hz (hardest part)
Week 7:  Jetson spidev + ROS2 Node     ──→  You can RECEIVE on the Jetson
Week 8:  EKF + Full Pipeline           ──→  You have a WORKING system
```

**Rule:** Finish each week's checkpoint before moving on. If you can't answer a checkpoint question from memory, re-read that section.

**Study approach:** Read the study notes → do the exercises → build on hardware (Week 3+). Each "day" assumes 3–4 hours of focused study. Adjust to your schedule.

### When You're Stuck (max 30 min per issue)

1. **Re-read the relevant study note section** (not the whole file — use Ctrl+F)
2. **Draw the problem** — timing diagram, signal flow, memory layout on paper
3. **Check the Day-Wasters list** in `zephyr/00-mastery-plan.md` Section 3
4. **Ask one specific question** — "GDB shows X but logic analyzer shows Y" beats "it doesn't work"
5. **If stuck >30 min:** Skip to the next section, mark it, come back tomorrow fresh

---

## All Materials at a Glance

```
learn/
├── STUDY-PLAN.md                          ← YOU ARE HERE
├── electronics/                           ← Week 1 + 2
│   ├── 00-learning-plan.md                   Topic dependency graph
│   ├── 01-passive-components.md              R, C, L, RLC (870 lines)
│   ├── 02-semiconductors.md                  Diodes, BJT, MOSFET (591 lines)
│   ├── 03-opamps-adc-sampling.md             Op-amps, ADC, Nyquist, MUX, ICs (856 lines)
│   ├── 04-uart-serial-deep-dive.md           Bit-level UART (564 lines)
│   ├── 05-spi-deep-dive.md                   Shift registers → frames (514 lines)
│   ├── 06-i2c-deep-dive.md                   Open-drain → transactions (571 lines)
│   ├── 07-can-deep-dive.md                   Differential → arbitration (724 lines)
│   └── exercises/
│       ├── 01-passive-components.md          R, C, L, RLC exercises
│       ├── 02-semiconductors.md              Diode, BJT, MOSFET exercises
│       ├── 03-opamps-adc-sampling.md         Op-amps, ADC, sampling exercises
│       ├── 04-uart-serial.md                 UART protocol exercises
│       ├── 05-spi.md                         SPI protocol exercises
│       ├── 06-i2c.md                         I2C protocol exercises
│       └── 07-can-bus.md                     CAN bus exercises
│
└── zephyr/                                ← Week 3–8
    ├── 00-mastery-plan.md                    11-project hardware plan
    ├── 01-17 source docs                     Reference material
    ├── study-notes/
    │   ├── 01-foundations.md                  Zephyr basics (915 lines)
    │   ├── 02-sensors.md                      I2C/CAN/UART Zephyr APIs (866 lines)
    │   ├── 03-zbus-nanopb.md                  Pub/sub + protobuf (1289 lines)
    │   ├── 04-spi-slave-dma.md                DMA double-buffer (1231 lines)
    │   └── 05-jetson-ros2.md                  spidev + ROS2 + EKF (1524 lines)
    └── exercises/
        ├── 01-foundations.md                  (1152 lines)
        ├── 02-sensors.md                      (1119 lines)
        ├── 03-zbus-nanopb.md                  (910 lines)
        ├── 04-spi-slave-dma.md                (633 lines)
        └── 05-jetson-ros2.md                  (649 lines)
```

**Total content: ~19,000+ lines across 27 files.**

Legend: 🎯 = essential for the SPI bridge goal | 📖 = skim-read OK | ⏭️ = skip if behind schedule

---

## Week 1 — Electronics Essentials (5 days)
*No hardware needed. Paper, pen, and a calculator.*

### Day 1: Resistors, Capacitors, RC Filters 🎯 (3.5 hrs)
**Read:** `electronics/01-passive-components.md` — Resistors + Capacitors + RLC sections
**Focus on:**
- V = IR — compute LED current-limiting resistors (you'll need this in Week 3)
- Voltage dividers — compute output voltage with and without a load
- Pull-up / pull-down — what "floating" means for CMOS inputs
- τ = RC time constant — the 63% rule and 5τ for "fully charged"
- Why 100nF decoupling caps go next to EVERY IC
- Low-pass RC filter: f₋₃dB = 1/(2πRC)

**Checkpoint — answer without looking:**
- [ ] 3.3V GPIO, LED Vf=2.0V, target 10mA. What resistor? What power does it dissipate?
- [ ] 10kΩ + 100nF filter. What's the cutoff frequency? (159 Hz)
- [ ] Why does a 100nF cap next to VCC help during a 10ns switching event but a 10µF doesn't?

---

### Day 2: Inductors, Diodes, Transistors 🎯 (3.5 hrs)
**Read:** `electronics/01-passive-components.md` — Inductors section
**Read:** `electronics/02-semiconductors.md` (full)
**Focus on:**
- Back-EMF when switching off inductive loads → flyback diode
- MOSFET as voltage-controlled switch — "logic-level" Vgs(th) < 3.3V
- Rds(on) determines heat. BSS138 level-shifting for 3.3V ↔ 5V I2C
- BJT: base current × β. MOSFET: Vgs > threshold. Know the difference.

**Checkpoint:**
- [ ] You switch off a relay coil. Why does a voltage spike appear?
- [ ] Why can't you use IRLZ44N (Vgs(th)=2-4V) reliably from 3.3V? (Threshold ≠ full enhancement)
- [ ] Draw the BSS138 level shifter from memory.

---

### Day 3: Op-Amps + ADC + Sampling Theory 🎯 (3.5 hrs)
**Read:** `electronics/03-opamps-adc-sampling.md` (full)
**Focus on:**
- Op-amp golden rules. Non-inverting gain = 1 + Rf/Rin. Voltage follower = buffer.
- 12-bit ADC: LSB = Vref/4096. SAR = binary search.
- Nyquist: sample ≥ 2× max frequency. Anti-alias filter BEFORE the ADC.
- Shift registers: the physical core of SPI TX/RX.

**Checkpoint:**
- [ ] Piezo 0–5V → your 0–3.3V ADC. Design the signal chain.
- [ ] 150Hz signal sampled at 200Hz. What do you see? (50Hz alias)
- [ ] What physically happens inside SPI TX on every clock edge? (Shift register shifts)

---

### Day 4: Exercises — Sections A, B, C only 🎯 (4 hrs)
**Do:** `electronics/exercises/01-passive-components.md` — Sections A, B, C
**Do:** `electronics/exercises/02-semiconductors.md` — Sections A, B, C
**Do:** `electronics/exercises/03-opamps-adc-sampling.md` — Sections A, B, C
- Re-read any section where you struggled
- Draw from memory: voltage divider, RC filter, BJT switch, MOSFET switch, non-inverting amp

---

### Day 5: Week 1 Review + Draw Everything (3 hrs)
Go through ALL Day 1–3 checkpoints from memory. Mark misses → re-read.

Draw from memory (paper, no peeking):
- [ ] RC low-pass filter with cutoff formula
- [ ] MOSFET low-side switch with gate resistor
- [ ] Non-inverting op-amp with gain formula
- [ ] Voltage divider → buffer → ADC chain

**You're ready for Week 2 when:** You can answer every checkpoint AND draw all 4 circuits.

> **Sections D & E** of the exercises are deeper-thinking problems. They're valuable but not blocking.
> ⏭️ Save them for evening/weekend review if time is tight.

---

## Week 2 — SPI-Focused Protocol Deep Dive (5 days)
*Theory + connect a logic analyzer if you have one.*

### Day 6: UART — Quick Essentials 📖 (2 hrs)
**Read:** `electronics/04-uart-serial-deep-dive.md` — skim (you use UART for Zephyr shell, but it's not the SPI bridge)
**Focus on the 3 things you'll actually need:**
- Bit-level frame: IDLE → START → 8 data → STOP. LSB first.
- Baud rate must match ±2%. 115200 is your debug default.
- DMA + idle-line interrupt for high-throughput (GPS in Week 4)

**Checkpoint:**
- [ ] Draw the waveform for 'A' (0x41) at 115200-8N1.
- [ ] Why is TX/RX always crossed?

---

### Day 7: SPI — The Shift Register Ring 🎯 (4 hrs)
**Read:** `electronics/05-spi-deep-dive.md` (full — read every word, this is your target protocol)
**Focus on:**
- Core insight: SPI = two shift registers in a ring. Always full-duplex.
- The 4 modes (CPOL × CPHA) — draw all 4 timing diagrams
- CS: active-low, pull-up during boot, one per slave
- Master vs slave timing asymmetry — why slave is hard
- Signal integrity: 40MHz SPI on 30cm wires fails (ringing, reflections)
- Protocol framing: magic byte + length + payload + CRC
- The "first byte garbage" problem in slave mode

**Checkpoint:**
- [ ] Master sends 0xA5 in Mode 0. Draw CLK, MOSI with data on each edge.
- [ ] Slave reads 0x52 when master sent 0xA5. What's wrong? (Wrong CPOL/CPHA)
- [ ] Why is SPI always full-duplex? (Shift register ring)
- [ ] Why does SPI need a framing protocol? (No message boundaries)

---

### Day 8: SPI Slave + DMA Concepts 🎯 (3.5 hrs)
**Do:** `electronics/exercises/05-spi.md` — all sections (while Day 7 is fresh)
**Read:** `zephyr/study-notes/04-spi-slave-dma.md` — PART 1 only (ELI15 concepts)
**Focus on:**
- Why DMA is necessary (CPU can't service SPI at 100Hz)
- Double buffering: Buffer A transmitting while Buffer B filled
- The pre-arming race: DMA must be ready BEFORE CS asserts
- D-cache coherency: **TX = Clean before DMA** (write-back dirty lines), **RX = Invalidate after DMA** (discard stale cache). Don't conflate them.
- `SCB_CleanDCache_by_Addr()` for TX, `SCB_InvalidateDCache_by_Addr()` for RX

**Checkpoint:**
- [ ] Why can't you arm DMA inside the CS ISR? (ISR latency > CS-to-clock gap)
- [ ] GDB shows correct TX buffer but Jetson reads garbage. What's wrong? (D-cache not cleaned before DMA TX)
- [ ] What's `__aligned(32)` for? (Cache line alignment — partial invalidate corrupts adjacent data)

---

### Day 9: I2C Essentials + CAN Overview 🎯📖 (3.5 hrs)
**Read:** `electronics/06-i2c-deep-dive.md` (full — you need I2C for the IMU in Week 4)
**Read:** `electronics/07-can-deep-dive.md` — **skim only** (CAN = 1 project in Week 4, not your main protocol)
**Focus on I2C:**
- Open-drain + pull-up. Wired-AND sharing.
- Complete transaction: START → addr+W → reg → RSTART → addr+R → data → NACK → STOP
- Bus stuck recovery: 9-clock bit-bang
**Skim CAN for:**
- Differential pair, dominant/recessive, 120Ω termination, frame format

**Checkpoint:**
- [ ] Why does I2C need pull-ups but SPI doesn't?
- [ ] Walk through reading register 0x75 from device 0x68: every byte on the wire
- [ ] CAN: what does a multimeter across CANH-CANL read with power off? (60Ω)

---

### Day 10: Protocol Exercises + Comparison Table (3.5 hrs)
**Do:** `electronics/exercises/04-uart-serial.md` — Sections A, B only ⏭️ (skip D, E)
**Do:** `electronics/exercises/06-i2c.md` — Sections A, B, C
**Do:** `electronics/exercises/07-can-bus.md` — Sections A, B only ⏭️ (skip D, E)

Fill in this table from memory, then check:

| Feature | UART | SPI | I2C | CAN |
|---------|------|-----|-----|-----|
| Wires | 2 (TX/RX) | 4+ (CLK/MOSI/MISO/CS) | 2 (SDA/SCL) | 2 (CANH/CANL) |
| Clock | Async | Master provides | Master provides | Embedded in data |
| Max speed | ~3 Mbaud | ~50 MHz | 3.4 MHz (HS) | 1 Mbps |
| Duplex | Full | Full | Half | Half |
| Best for | Debug/GPS | High-speed chip-to-chip | Config/sensors | Automotive |

**You're ready for Week 3 when:** You can draw the SPI timing diagram for all 4 modes AND the I2C read transaction.

---

### 🔄 RECALL CHECKPOINT #1 (end of Week 2 — 15 min)
Without opening any file, answer these from Weeks 1–2:
- [ ] Calculate: 3.3V, LED Vf=2.0V, 10mA. Resistor = ? Power = ?
- [ ] Draw: MOSFET low-side switch
- [ ] Draw: SPI waveform for one byte in Mode 0 (CPOL=0, CPHA=0) — show SCLK, MOSI, CS

*If you miss ≥2: spend 30 min reviewing the relevant sections before starting Week 3.*

---

## Week 3 — Zephyr Foundations (5 days)
*Hardware required: STM32 Nucleo + logic analyzer*

### Day 11–12: Blinky + Shell + 100Hz Timer 🎯 (7 hrs)
**Read:** `zephyr/study-notes/01-foundations.md` (full, 915 lines)
**Do:** `zephyr/exercises/01-foundations.md` — Sections A through C
**Build:** Projects 1, 2, 3 from `zephyr/00-mastery-plan.md`
**Verify:** LED blinks, shell responds, `dt_ms=10` logged at 100Hz

**Checkpoint:**
- [ ] Write a minimal `CMakeLists.txt` + `prj.conf` + `main.c` for Zephyr blinky from memory
- [ ] `k_msleep(10)` vs `k_timer` for 100Hz — which drifts and why?
- [ ] Default stack is 1KB. Thread uses LOG_INF. What happens? (Stack overflow, silent death)

---

### Day 13–15: Foundation Exercises + Hardening (7 hrs)
**Do:** `zephyr/exercises/01-foundations.md` — Sections D + E
**Focus on:**
- Thread stack sizing (use `CONFIG_THREAD_ANALYZER`)
- Watchdog timer setup (IWDG — add `CONFIG_WDT=y`, feed in main loop, system resets on hang)
- Zephyr logging levels and `CONFIG_LOG_MODE_DEFERRED` for real-time threads

**Checkpoint:**
- [ ] How do you detect stack overflow in Zephyr? (CONFIG_THREAD_ANALYZER or CONFIG_MPU_STACK_GUARD)
- [ ] What happens if your 100Hz thread takes 12ms? (Misses deadline, timer fires again immediately)

---

## Week 4 — Sensors + ZBus + nanopb (5 days)
*Projects 4–7 in one week — tight but doable because theory is done*

### Day 16–17: I2C IMU + CAN Encoder 🎯 (7 hrs)
**Read:** `zephyr/study-notes/02-sensors.md` — I2C + CAN sections
**Do:** `zephyr/exercises/02-sensors.md` — Sections A, B, C
**Build:** Project 4 (ICM-42688 over I2C at 100Hz)
**Build:** Project 5 (CAN receive from second node)
**Verify:** `acc_z ≈ 9.81` flat. Logic analyzer shows clean I2C. `candump` shows CAN frames.

---

### Day 18: UART GPS 📖 (3 hrs)
**Read:** `zephyr/study-notes/02-sensors.md` — UART section only (skim I2C/CAN parts)
**Build:** Project 6 (NMEA parser with ring buffer)
**Verify:** Parses `$GNGGA` correctly. Handles "no fix" without crashing.

---

### Day 19–20: ZBus + nanopb 🎯 (7 hrs)
**Read:** `zephyr/study-notes/03-zbus-nanopb.md` (full, 1289 lines)
**Do:** `zephyr/exercises/03-zbus-nanopb.md` — Sections A, B, C
**Build:** Project 7 (ZBus publish all sensors → nanopb encode → round-trip verify)
**Verify:** `encoded 47 bytes` at 100Hz, size always identical, round-trip decode matches

**Checkpoint:**
- [ ] proto3 all-zero message — what happens to the size? (Fewer bytes, fields omitted including `payload_length` if zero)
- [ ] ZBus subscriber vs listener — when to use each?
- [ ] What protects ZBus channel data from races? (Mutex)

---

### 🔄 RECALL CHECKPOINT #2 (end of Week 4 — 15 min)
Without opening any file, answer from Weeks 1–2:
- [ ] Draw SPI Mode 0 timing for 0xA5 (CLK, MOSI, CS)
- [ ] What is D-cache clean vs invalidate? Which for TX? Which for RX?
- [ ] I2C: read register 0x75 from device 0x68 — all bytes on the wire

*Miss ≥2: review before the hardest part (Week 5).*

---

## Week 5–6 — SPI Slave DMA (10 days)
*This is the hardest project. Budget 2 full weeks.*

### Day 21–22: Theory + DMA Deep Dive 🎯 (7 hrs)
**Read:** `zephyr/study-notes/04-spi-slave-dma.md` (full, 1231 lines — all of it)
**Do:** `zephyr/exercises/04-spi-slave-dma.md` — Sections A through C

---

### Day 23–28: Build + Debug + Iterate 🎯 (18 hrs)
**Build:** Project 8 (SPI slave DMA at 100Hz with double-buffering)

**Debug checklist — check these IN ORDER when things break:**
1. Logic analyzer: is CS asserting at 100Hz? Is CLK present during CS?
2. Is DMA armed BEFORE CS? (Check pre-arm timing with GPIO toggle)
3. D-cache: TX cleaned? RX invalidated? Buffer `__aligned(32)` and in D2 SRAM?
4. Is the `slot` / double-buffer swap atomic? (ISR context vs thread context)
5. nanopb: does decoded output match? (Round-trip test)

**Verify with logic analyzer:**
- [ ] 100 CS assertions/sec, evenly spaced (±0.5ms jitter max)
- [ ] No garbage first-bytes (pre-arming race solved)
- [ ] nanopb decoded correctly by Python spidev script on second board
- [ ] Latency from CS assert to first clock edge < 5µs

---

### Day 29–30: Exercises + Edge Cases (6 hrs)
**Do:** `zephyr/exercises/04-spi-slave-dma.md` — Sections D (lab) + E (timing math)

**Checkpoint:**
- [ ] Why does arming DMA in the CS ISR fail? (ISR latency > CS-to-clock gap)
- [ ] GDB shows correct buffer, Jetson reads garbage. Diagnosis? (D-cache not cleaned for TX)
- [ ] DMA buffer in DTCM — works? (No. STM32H7 DMA can't access DTCM. Must be D2 SRAM.)
- [ ] What happens if master polls faster than firmware can swap buffers? (Stale data — same frame twice)

---

## Week 7 — Jetson spidev + ROS2 Node (5 days)
*Hardware: Jetson Orin + STM32 connected via SPI*

### Day 31–32: Python spidev Reader 🎯 (6 hrs)
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 1 sections 1–6
**Build:** Project 9 (Python spidev on Jetson reads + decodes at 100Hz)
**Verify:** `sudo spidev_test` works first, then Python shows values matching STM32

---

### Day 33–35: ROS2 Publisher Node 🎯 (9 hrs)
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 1 sections 7–15
**Do:** `zephyr/exercises/05-jetson-ros2.md` — Sections A, B, C
**Build:** Project 10 (rclpy node publishing Imu, Odometry, NavSatFix at 100Hz)
**Verify:** `ros2 topic hz /imu` → ~100Hz. rviz2 shows rotating IMU axes.

**Checkpoint:**
- [ ] `xfer2()` in timer callback → 87Hz. Why? (Blocking SPI call + timer jitter)
- [ ] QoS for sensor topics? (BEST_EFFORT — RELIABLE backpressure kills 100Hz)
- [ ] Timestamps: use ROS clock (`self.get_clock().now()`), NOT `CLOCK_MONOTONIC_RAW`

---

## Week 8 — EKF + Full Pipeline Validation (5 days)
*The finish line: end-to-end 100Hz sensor fusion*

### Day 36–37: robot_localization EKF 🎯 (7 hrs)
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 2 (code) + PART 3 (gotchas)
**Do:** `zephyr/exercises/05-jetson-ros2.md` — Sections D (lab) + E (system design)
**Build:** Project 11 (robot_localization EKF, TF2 frame tree, rviz2 visualization)

---

### Day 38–39: Full System Validation 🎯 (6 hrs)
**Verify the complete pipeline — all of these must pass:**
- [ ] `ros2 topic hz /odometry/filtered` → 100Hz
- [ ] `ros2 run tf2_tools view_frames` → complete `map→odom→base_link→imu_frame` tree
- [ ] Push board sideways → EKF estimates velocity correctly
- [ ] Logic analyzer: SPI CS at 100Hz ±0.5ms, clean clocking, no first-byte garbage
- [ ] Run for 10 minutes continuously — no drift, no crashes, no dropped frames

---

### Day 40: Retrospective + What's Next (3 hrs)
- [ ] Review your notes: what was hardest? What would you do differently?
- [ ] Document your setup: wiring diagram, Zephyr config, ROS2 launch file
- [ ] Identify gaps for production: error recovery, power management, OTA updates

---

### 🔄 FINAL RECALL CHECKPOINT (15 min)
Answer all of these without any files open:
- [ ] Draw the full SPI slave DMA double-buffer data flow
- [ ] What's the D-cache rule? (TX: clean before. RX: invalidate after.)
- [ ] Why D2 SRAM, not DTCM?
- [ ] I2C read transaction: all bytes for reading reg 0x75 from 0x68
- [ ] Voltage divider 10k+10k on 5V with 1kΩ load: output = ?
- [ ] EKF diverges. First thing to check? (Timestamps)

---

## Summary: What You Can Do After Each Phase

| After | You Can... |
|-------|-----------|
| Week 1–2 | Read a schematic, calculate R/C values, draw SPI/I2C timing, understand D-cache coherency |
| Week 3–4 | Build Zephyr firmware reading real sensors at 100Hz, encode with nanopb |
| Week 5–6 | Run SPI slave DMA at 100Hz with double-buffering — the hardest embedded skill |
| Week 7–8 | Full pipeline: STM32 → SPI → Jetson → ROS2 → EKF at 100Hz |

---

## What Was Trimmed (and where to find it later)

| Trimmed | Why | Where to find it |
|---------|-----|-------------------|
| Electronics Sections D, E exercises | Deeper thinking — valuable but not blocking | `electronics/exercises/*.md` Sections D, E |
| UART deep exercises | You already know serial from ROS2/Python | `electronics/exercises/04-uart-serial.md` D, E |
| CAN deep exercises | CAN = 1 project, not your main protocol | `electronics/exercises/07-can-bus.md` D, E |
| Sensor exercises D, E | Deeper sensor analysis — not blocking | `zephyr/exercises/02-sensors.md` Sections D, E |
| ZBus + nanopb exercises D, E | Advanced encoding patterns — not blocking | `zephyr/exercises/03-zbus-nanopb.md` Sections D, E |
| C++ Advanced track | Separate 8-week curriculum, run in parallel or after | `cpp-advanced/STUDY-PLAN.md` |

---

## Quick Links to Key Reference Material

| When you need... | Go to... |
|------------------|----------|
| Resistor/cap/inductor formulas | `electronics/01-passive-components.md` → Quick Reference Card |
| MOSFET switch circuit | `electronics/02-semiconductors.md` → MOSFET section |
| Op-amp gain formulas | `electronics/03-opamps-adc-sampling.md` → Quick Reference Card |
| UART waveform format | `electronics/04-uart-serial-deep-dive.md` → Bit-level protocol |
| SPI mode timing diagrams | `electronics/05-spi-deep-dive.md` → 4 SPI modes |
| I2C pull-up calculation | `electronics/06-i2c-deep-dive.md` → Pull-up value section |
| CAN arbitration rules | `electronics/07-can-deep-dive.md` → Arbitration section |
| Zephyr gotcha tables | `zephyr/study-notes/0X-*.md` → PART 3 in each file |
| Hardware wiring safety | `zephyr/00-mastery-plan.md` → Section 2 |
| Day-waster failure modes | `zephyr/00-mastery-plan.md` → Section 3 |
