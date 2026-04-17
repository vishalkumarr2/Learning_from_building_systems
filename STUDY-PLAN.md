# Unified Study Plan: Electronics → Protocols → Embedded Systems
### From resistors to a 100Hz STM32 ↔ Jetson SPI bridge
### Total: ~110–140 hours across 4 phases

---

## How This Plan Works

```
Phase 1: Electronics Foundations     ──→  You understand the PHYSICAL layer
Phase 2: Protocol Deep Dives        ──→  You understand the WIRE layer
Phase 3: Zephyr + STM32 Projects    ──→  You can BUILD firmware
Phase 4: Jetson + ROS2 Integration  ──→  You have a WORKING system
```

**Rule:** Finish each week's checkpoint before moving on. If you can't answer a checkpoint question from memory, re-read that section.

**Study approach:** Read the study notes → do the exercises → build on hardware (Phase 3+). Each "day" assumes 3–4 hours of focused study. Adjust to your schedule.

---

## All Materials at a Glance

```
learn/
├── STUDY-PLAN.md                          ← YOU ARE HERE
├── electronics/                           ← Phase 1 + 2
│   ├── 00-learning-plan.md                   Topic dependency graph
│   ├── 01-passive-components.md              R, C, L, RLC (870 lines)
│   ├── 02-semiconductors.md                  Diodes, BJT, MOSFET (591 lines)
│   ├── 03-opamps-adc-sampling.md             Op-amps, ADC, Nyquist, MUX, ICs (856 lines)
│   ├── 04-uart-serial-deep-dive.md           Bit-level UART (564 lines)
│   ├── 05-spi-deep-dive.md                   Shift registers → frames (514 lines)
│   ├── 06-i2c-deep-dive.md                   Open-drain → transactions (571 lines)
│   ├── 07-can-deep-dive.md                   Differential → arbitration (724 lines)
│   └── exercises/                         ← NEW: Phase 1 + 2 exercises
│       ├── 01-passive-components.md          R, C, L, RLC exercises
│       ├── 02-semiconductors.md              Diode, BJT, MOSFET exercises
│       ├── 03-opamps-adc-sampling.md         Op-amps, ADC, sampling exercises
│       ├── 04-uart-serial.md                 UART protocol exercises
│       ├── 05-spi.md                         SPI protocol exercises
│       ├── 06-i2c.md                         I2C protocol exercises
│       └── 07-can-bus.md                     CAN bus exercises
│
└── zephyr/                                ← Phase 3 + 4
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

---

## Phase 1 — Electronics Foundations (Week 1–2)
*No hardware needed. Paper, pen, and a calculator.*

### Week 1: Passive Components + Semiconductors

#### Day 1: Resistors & Ohm's Law (3 hrs)
**Read:** `electronics/01-passive-components.md` — Resistors section only
**Focus on:**
- V = IR — compute LED current-limiting resistors (you'll need this in Week 5)
- Voltage dividers — compute output voltage with and without a load
- Pull-up / pull-down — what "floating" means for CMOS inputs
- Power dissipation P = I²R — when a resistor gets dangerously hot

**Checkpoint — answer without looking:**
- [ ] 3.3V GPIO, LED Vf=2.0V, target 10mA. What resistor? What power does it dissipate?
- [ ] 10kΩ + 10kΩ divider on 5V. Load is 1kΩ. What's the actual output voltage?
- [ ] Why does an unconnected CMOS input oscillate randomly?

---

#### Day 2: Capacitors & RC Circuits (3 hrs)
**Read:** `electronics/01-passive-components.md` — Capacitors + RLC sections
**Focus on:**
- The water tank analogy for capacitor charging
- τ = RC time constant — the 63% rule and 5τ for "fully charged"
- Xc = 1/(2πfC) — why caps block DC, pass AC
- Why 100nF decoupling caps go next to EVERY IC
- Low-pass RC filter: f₋₃dB = 1/(2πRC)

**Checkpoint:**
- [ ] 10kΩ + 100nF filter. What's the cutoff frequency? (159 Hz)
- [ ] Why does a 100nF cap next to VCC help during a 10ns digital switching event but a 10µF electrolytic doesn't?
- [ ] How long to charge a 1µF cap through 10kΩ to 99% of Vcc? (5 × 10ms = 50ms)

---

#### Day 3: Inductors + Diodes (3 hrs)
**Read:** `electronics/01-passive-components.md` — Inductors section
**Read:** `electronics/02-semiconductors.md` — Diodes section
**Focus on:**
- Inductor = flywheel analogy (resists current changes)
- Back-EMF when switching off inductive loads → flyback diode
- PN junction, forward drop (0.7V Si, 0.3V Schottky)
- Zener diodes for voltage clamping / regulation
- TVS diodes for ESD protection on signal lines

**Checkpoint:**
- [ ] You switch off a relay coil. Why does a voltage spike appear and how does a flyback diode fix it?
- [ ] A 3.3V GPIO drives an LED through 150Ω. The LED Vf is 1.8V. What current flows? (10mA)
- [ ] Why are Schottky diodes used instead of regular silicon in power supply rectification?

---

#### Day 4: Transistors — BJT + MOSFET (4 hrs)
**Read:** `electronics/02-semiconductors.md` — BJT + MOSFET sections (full)
**Focus on:**
- BJT as a current-controlled switch (base current × β = collector current)
- MOSFET as a voltage-controlled switch (Vgs > Vgs(th) → channel opens)
- "Logic-level" MOSFET — why Vgs(th) must be < 3.3V for MCU-driven switching
- Rds(on) — on-resistance determines heat in power switching
- BSS138 level-shifting circuit for I2C 3.3V ↔ 5V
- H-bridge concept (4 MOSFETs for DC motor direction control)

**Checkpoint:**
- [ ] Design a circuit: 3.3V GPIO → NPN BJT → 12V/50mA relay coil. Calculate base resistor.
- [ ] Why can't you use a standard IRLZ44N (Vgs(th)=2-4V) reliably from 3.3V GPIO? (Threshold is *start* of conduction, not full enhancement)
- [ ] Draw the BSS138 level shifter. How does it shift bidirectionally?
- [ ] BJT vs MOSFET: which needs continuous base/gate current to stay ON?

---

#### Day 5: Exercises + Review (4 hrs)
**Do:** `electronics/exercises/01-passive-components.md` — Sections A, B, C (conceptual + spot-the-bug + fill-in)
**Do:** `electronics/exercises/02-semiconductors.md` — Sections A, B, C
- Re-read any section where you couldn't answer
- Work through the comparison tables in `01-passive-components.md` and `02-semiconductors.md`
- Draw from memory: voltage divider, RC filter, BJT switch circuit, MOSFET switch circuit, BSS138 level shifter

---

### Week 2: Op-Amps, ADC, Sampling + Digital Building Blocks

#### Day 6: Op-Amps (3 hrs)
**Read:** `electronics/03-opamps-adc-sampling.md` — Op-Amp section
**Focus on:**
- Two golden rules (with negative feedback): no input current, inputs at same voltage
- Inverting amp: gain = −Rf/Rin
- Non-inverting amp: gain = 1 + Rf/Rin
- Voltage follower (buffer): WHY high-impedance sensors need one before ADC
- Comparator: op-amp without feedback, output slams to rail

**Checkpoint:**
- [ ] Non-inverting amp with Rf=10kΩ, Rin=2kΩ. Gain? (6)
- [ ] A piezo sensor outputs 0–5V but your ADC takes 0–3.3V. Design the signal chain.
- [ ] What happens if you forget negative feedback on an op-amp? (Becomes a comparator)

---

#### Day 7: ADC + Sampling Theory (3 hrs)
**Read:** `electronics/03-opamps-adc-sampling.md` — ADC + Sampling Theory sections
**Focus on:**
- 12-bit ADC: LSB = Vref / 4096 — for 3.3V, one step = 0.806mV
- SAR ADC (binary search, one bit per clock — most common in MCUs)
- Nyquist theorem: sample at ≥ 2× highest frequency. Violate it → aliasing.
- Aliasing: the "wagon wheel" effect. 150Hz signal sampled at 200Hz → appears as 50Hz
- Anti-aliasing filter MUST come BEFORE the ADC (physics, not fixable in software)
- Oversampling: sample at 10× and average → ~1.5 extra effective bits

**Checkpoint:**
- [ ] 12-bit ADC reads 2048 with Vref=3.3V. What voltage? (1.65V)
- [ ] You sample a 150Hz signal at 200Hz. What frequency do you see? (50Hz — alias)
- [ ] Why can't you fix aliasing after digitization?
- [ ] What order of RC filter cutoff do you need before a 1kHz-sampled ADC? (<500Hz)

---

#### Day 8: MUX, Serializer, Deserializer, ICs (2 hrs)
**Read:** `electronics/03-opamps-adc-sampling.md` — Digital Building Blocks section
**Focus on:**
- Multiplexer: N inputs → 1 output, selected by address bits
- Serializer (parallel → serial): the core of SPI TX
- Deserializer (serial → parallel): the core of SPI RX
- Shift registers: 74HC595 (output expansion), 74HC165 (input expansion)
- IC packages: DIP, SOIC, QFP, BGA — how to read pin 1
- How to read page 1 of a datasheet (max voltage, max current, key parameter)

**Checkpoint:**
- [ ] You have 8 LEDs but only 3 GPIO pins. How do you control all 8? (74HC595 shift register)
- [ ] What physically happens inside SPI TX on every clock edge? (Shift register shifts one bit out, one bit in)

---

#### Day 9: Exercises + Phase 1 Comprehensive Review (4 hrs)
**Do:** `electronics/exercises/01-passive-components.md` — Sections D, E (lab/calc + deeper thinking)
**Do:** `electronics/exercises/02-semiconductors.md` — Sections D, E
**Do:** `electronics/exercises/03-opamps-adc-sampling.md` — all sections

Go through ALL checkpoint questions from Days 1–8. Mark any you can't answer → re-read.

Draw from memory:
- [ ] RC low-pass filter with cutoff calculation
- [ ] NPN BJT switch circuit with base resistor
- [ ] MOSFET low-side switch
- [ ] Non-inverting op-amp with gain formula
- [ ] Voltage divider → buffer → ADC chain

**You're ready for Phase 2 when:** You can answer every checkpoint AND complete all Section D exercises from memory.

---

## Phase 2 — Protocol Deep Dives (Week 3–4)
*Still mostly theory, but connect a logic analyzer if you have one.*

### Week 3: UART + SPI

#### Day 10: UART from Electrons to Bytes (3 hrs)
**Read:** `electronics/04-uart-serial-deep-dive.md` (full)
**Focus on:**
- Bit-level frame: IDLE → START (falling edge) → 8 data bits (LSB first!) → STOP
- Draw the waveform for transmitting 'U' (0x55) — alternating 0/1 pattern
- Baud rate = bit-times per second. Both sides must match within ±2%
- RS-232 (±12V) vs TTL (0/3.3V) — why level shifters exist
- DMA + idle-line interrupt for high-speed UART
- Ring buffers: why interrupt-per-byte fails at high baud rates

**Checkpoint:**
- [ ] Draw the complete waveform for 'A' (0x41) at 115200-8N1. How many microseconds total?
- [ ] Two devices at 115200 and 112000. What's the % error? At which byte does corruption start?
- [ ] Why is TX and RX always crossed? (TX of device A → RX of device B)
- [ ] What's the advantage of RS-485 over plain UART? (Differential, multi-drop, longer distance)

---

#### Day 11: SPI — The Shift Register Ring (4 hrs)
**Read:** `electronics/05-spi-deep-dive.md` (full)
**Focus on:**
- Core insight: SPI = two shift registers in a ring. Always full-duplex. Always exchanging data.
- The 4 modes (CPOL × CPHA) — draw all 4 timing diagrams
- CS: active-low, pull-up during boot, one per slave
- SPI master vs slave timing asymmetry — why slave is hard
- Signal integrity: why 40MHz SPI on 30cm wires fails (ringing)
- Protocol framing: magic byte + length + payload + CRC
- The "first byte garbage" problem in slave mode

**Checkpoint:**
- [ ] Master sends 0xA5 in Mode 0. Draw CLK, MOSI with data on each edge.
- [ ] SPI slave reads 0x52 when master sent 0xA5. What's wrong? (Wrong CPOL/CPHA — bits shifted by 1)
- [ ] Why is SPI always full-duplex even when you only want to read? (Shift register ring — master must clock out dummy bytes)
- [ ] Why does SPI need a framing protocol on top? (No inherent message boundaries)

---

#### Day 12: SPI Slave + DMA — The Hard Part (4 hrs)
**Do:** `electronics/exercises/04-uart-serial.md` — all sections (while Day 10 is fresh)
**Do:** `electronics/exercises/05-spi.md` — all sections (while Day 11 is fresh)
**Read:** `zephyr/study-notes/04-spi-slave-dma.md` — PART 1 only (ELI15 concepts)
**Focus on:**
- Why DMA is necessary (CPU can't service SPI fast enough at 100Hz)
- Double buffering: Buffer A transmitting while Buffer B being filled
- The pre-arming race: DMA must be ready BEFORE CS asserts
- D-cache coherency: GDB lies to you (cache shows correct, DMA sees stale)
- `SCB_CleanInvalidateDCache_by_Addr()` — clean = write back, invalidate = mark stale

**Checkpoint:**
- [ ] Why can't you arm DMA inside the CS assert interrupt? (Interrupt latency ~1-10µs, CS-to-clock ~100ns)
- [ ] GDB shows your TX buffer is correct but the Jetson reads garbage. What's wrong? (D-cache not flushed before DMA)
- [ ] What does `__aligned(32)` do on a buffer and why does DMA need it?

---

### Week 4: I2C + CAN

#### Day 13: I2C — Open Drain to Transactions (4 hrs)
**Read:** `electronics/06-i2c-deep-dive.md` (full)
**Focus on:**
- Open-drain: MOSFET only pulls LOW, pull-up resistor pulls HIGH. This IS the key to I2C.
- Wired-AND: any device pulling low = line is low. Multiple devices share safely.
- Pull-up resistor calculation: R < rise_time / (0.8473 × C_bus)
- Complete transaction walkthrough: START → address+W → register → REPEATED START → address+R → data → NACK → STOP
- Bus stuck recovery: 9-clock bit-bang (slave stuck mid-byte, SDA held low)
- Clock stretching: slave holds SCL low to buy time

**Checkpoint:**
- [ ] Why does I2C need pull-ups but SPI doesn't? (Open-drain vs push-pull)
- [ ] Calculate pull-up for 400kHz I2C, 200pF bus capacitance. (< 1.77kΩ, use 2.2kΩ)
- [ ] Walk through reading register 0x75 from device 0x68: every byte on the wire with R/W bit
- [ ] Bus is stuck with SDA low. Why does clocking SCL 9 times fix it?

---

#### Day 14: CAN — Differential Signaling + Arbitration (3 hrs)
**Read:** `electronics/07-can-deep-dive.md` (full)
**Focus on:**
- Differential pair: CANH−CANL. Noise on both wires cancels out.
- Dominant (0) vs Recessive (1): dominant always wins → wired-AND on the bus
- 120Ω termination at each end (60Ω total). Without it: reflections cause bit errors.
- Frame format: SOF → ID → RTR → DLC → Data → CRC → ACK → EOF
- Arbitration: lower ID wins because dominant=0. Bit-by-bit comparison during ID field.
- Error states: Error Active → Error Passive → Bus Off
- Bit-stuffing: insert opposite bit after 5 consecutive same bits (for clock recovery)

**Checkpoint:**
- [ ] Node A sends ID 0x100, Node B sends ID 0x080 simultaneously. Who wins and at which bit?
- [ ] Multimeter across CANH-CANL with power off. What should it read? (60Ω with both terminators)
- [ ] What happens to a node that keeps getting CRC errors? (TEC/REC increment → Bus Off)
- [ ] Why does CAN need bit-stuffing? (Receiver recovers clock from edges; 5+ same bits = no edge = clock drift)

---

#### Day 15: Exercises + Protocol Comparison + Phase 2 Review (4 hrs)
**Do:** `electronics/exercises/06-i2c.md` — all sections (while Day 13 is fresh)
**Do:** `electronics/exercises/07-can-bus.md` — all sections (while Day 14 is fresh)

Fill in this table from memory, then check against the deep-dive docs:

| Feature | UART | SPI | I2C | CAN |
|---------|------|-----|-----|-----|
| Wires | ? | ? | ? | ? |
| Clock | ? | ? | ? | ? |
| Max speed | ? | ? | ? | ? |
| Max distance | ? | ? | ? | ? |
| Multi-device? | ? | ? | ? | ? |
| Duplex | ? | ? | ? | ? |
| Best for | ? | ? | ? | ? |

**Answers:**
| Feature | UART | SPI | I2C | CAN |
|---------|------|-----|-----|-----|
| Wires | 2 (TX/RX) | 4+ (CLK/MOSI/MISO/CS) | 2 (SDA/SCL) | 2 (CANH/CANL) |
| Clock | None (async) | Master provides | Master provides | Embedded in data |
| Max speed | ~3 Mbaud | ~50 MHz | 3.4 MHz (HS) | 1 Mbps (classic) |
| Max distance | ~15m (RS-232) | ~30cm at speed | ~1m at 400kHz | ~500m at 125kbps |
| Multi-device? | Point-to-point | 1 master, N slaves (via CS) | Multi-master, N slaves (via address) | Multi-master, broadcast |
| Duplex | Full | Full | Half | Half |
| Best for | Debug console, GPS | High-speed chip-to-chip | Config registers, sensors | Automotive, industrial |

Re-do all Week 3–4 checkpoints from memory.

**You're ready for Phase 3 when:** You can draw the timing diagram for ANY of the 4 protocols from memory, and you can calculate I2C pull-up values.

---

## Phase 3 — Zephyr + STM32 Hands-On (Week 5–9)
*Hardware required: STM32 Nucleo + logic analyzer + sensors*

### Week 5: Zephyr Toolchain + Foundations (Projects 1–3)

#### Day 16–17: Blinky + Shell + 100Hz Timer
**Read:** `zephyr/study-notes/01-foundations.md` (full, 915 lines)
**Do:** `zephyr/exercises/01-foundations.md` — Sections A through C
**Build:** Projects 1, 2, 3 from `zephyr/00-mastery-plan.md`
**Verify:** LED blinks, shell responds, `dt_ms=10` logged at 100Hz

**Checkpoint:**
- [ ] From memory: write a minimal `CMakeLists.txt`, `prj.conf`, and `main.c` for Zephyr blinky
- [ ] What's the difference between `k_msleep(10)` and `k_timer` for 100Hz? (msleep drifts, timer doesn't)
- [ ] Default stack is 1KB. Your thread uses LOG_INF. What happens? (Stack overflow, silent death)

---

### Week 6: Sensor Integration (Projects 4–6)

#### Day 18–19: I2C IMU Read
**Read:** `zephyr/study-notes/02-sensors.md` — I2C sections
**Do:** `zephyr/exercises/02-sensors.md` — Sections A through C (I2C questions)
**Build:** Project 4 (ICM-42688 over I2C at 100Hz)
**Verify:** `acc_z ≈ 9.81` when board is flat. Logic analyzer shows clean I2C transactions.

#### Day 20: CAN Encoder
**Read:** `zephyr/study-notes/02-sensors.md` — CAN sections
**Build:** Project 5 (CAN receive from second node)
**Verify:** `candump` on USB-CAN shows frames. Zephyr prints parsed velocity.

#### Day 21: UART GPS
**Read:** `zephyr/study-notes/02-sensors.md` — UART sections
**Build:** Project 6 (NMEA parser with ring buffer)
**Verify:** Parses `$GNGGA` correctly. Handles indoor "no fix" without crashing.

#### Day 22: Sensor exercises
**Do:** `zephyr/exercises/02-sensors.md` — Sections D (lab tasks) + E (deeper thinking)

---

### Week 7: ZBus + nanopb (Project 7)

#### Day 23–24: Pub/Sub + Protobuf Encoding
**Read:** `zephyr/study-notes/03-zbus-nanopb.md` (full, 1289 lines)
**Do:** `zephyr/exercises/03-zbus-nanopb.md` — all sections
**Build:** Project 7 (ZBus publish all sensors → nanopb encode → round-trip verify)
**Verify:** `encoded 47 bytes` at 100Hz, size always identical, round-trip decode matches

**Checkpoint:**
- [ ] What happens if proto3 encodes an all-zero message? (Fewer bytes than max — frame size varies)
- [ ] ZBus subscriber vs listener — when to use each?
- [ ] What does ZBus internally do to protect the channel data? (Mutex around the channel struct)

---

### Week 8–9: SPI Slave DMA (Project 8)
*Budget 2 weeks — this is the hardest project*

#### Day 25–26: Theory + Code
**Read:** `zephyr/study-notes/04-spi-slave-dma.md` (full, 1231 lines)
**Do:** `zephyr/exercises/04-spi-slave-dma.md` — Sections A through C

#### Day 27–30: Build + Debug
**Build:** Project 8 (SPI slave DMA at 100Hz with double-buffering)
**Verify with logic analyzer:**
- [ ] 100 CS assertions per second, evenly spaced
- [ ] No garbage first-bytes (pre-arming race solved)
- [ ] nanopb decoded correctly by a Python spidev script on second board

**Do:** `zephyr/exercises/04-spi-slave-dma.md` — Sections D (lab) + E (timing math)

**Checkpoint:**
- [ ] Why does arming DMA in the CS ISR fail? (ISR latency > CS-to-clock gap)
- [ ] Code compiles, GDB shows correct buffer, Jetson reads garbage. What do you check? (D-cache flush)
- [ ] What memory region must the DMA buffer be in on STM32H7? (D2 SRAM, not DTCM)

---

## Phase 4 — Jetson + ROS2 Integration (Week 10–12)
*Hardware required: Jetson Orin + STM32 connected via SPI*

### Week 10: spidev + Decode (Project 9)

#### Day 31–32: Python SPI Reader
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 1 sections 1–6
**Build:** Project 9 (Python spidev on Jetson reads + decodes at 100Hz)
**Verify:** `sudo spidev_test` works first, then Python shows sensor values matching STM32

---

### Week 11: ROS2 Publisher (Project 10)

#### Day 33–34: ROS2 Node
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 1 sections 7–15
**Do:** `zephyr/exercises/05-jetson-ros2.md` — Sections A through C
**Build:** Project 10 (rclpy node publishing Imu, Odometry, NavSatFix at 100Hz)
**Verify:** `ros2 topic hz /imu` → ~100Hz. rviz2 shows rotating IMU axes.

**Checkpoint:**
- [ ] Why does putting `xfer2()` in the timer callback cause 87Hz instead of 100Hz?
- [ ] What QoS profile for sensor topics and why? (BEST_EFFORT — RELIABLE causes backpressure at 100Hz)
- [ ] Your EKF diverges. What do you check first? (Timestamp accuracy — must be captured immediately after ioctl)

---

### Week 12: EKF + Full Pipeline (Project 11)

#### Day 35–37: robot_localization + Validation
**Read:** `zephyr/study-notes/05-jetson-ros2.md` — PART 2 (code) + PART 3 (gotchas)
**Do:** `zephyr/exercises/05-jetson-ros2.md` — Sections D (lab) + E (system design)
**Build:** Project 11 (robot_localization EKF, TF2 frame tree, rviz2 visualization)
**Verify:**
- [ ] `ros2 topic hz /odometry/filtered` → 100Hz
- [ ] `ros2 run tf2_tools view_frames` → complete `map→odom→base_link→imu_frame` tree
- [ ] Push the board sideways → EKF estimates velocity correctly

---

## Summary: What You Can Do After Each Phase

| After | You Can... |
|-------|-----------|
| Phase 1 | Read a schematic, calculate resistor/cap values, understand transistor circuits, design signal conditioning |
| Phase 2 | Read any SPI/I2C/CAN/UART waveform on a logic analyzer, calculate timing budgets, debug protocol issues from the wire level |
| Phase 3 | Build Zephyr firmware that reads real sensors at 100Hz, encodes protobuf, and DMA-transfers over SPI slave |
| Phase 4 | Integrate with ROS2 on Jetson, fuse sensor data in EKF, visualize in rviz2 — end-to-end working pipeline |

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
| Electronics practice problems | `electronics/exercises/01-07*.md` → Sections A–E per chapter |
| Zephyr gotcha tables | `zephyr/study-notes/0X-*.md` → PART 3 in each file |
| Zephyr practice problems | `zephyr/exercises/0X-*.md` → Sections A–E per chapter |
| Hardware wiring safety | `zephyr/00-mastery-plan.md` → Section 2 |
| Day-waster failure modes | `zephyr/00-mastery-plan.md` → Section 3 |
