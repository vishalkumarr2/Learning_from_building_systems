# 05 — SPI Deep Dive
### From shift registers to frames: the protocol YOUR bridge uses
**Prerequisites:** Chapter 01 (RC, decoupling), Chapter 02 (MOSFETs)
**Unlocks:** 100Hz STM32↔Jetson bridge, IMU reads, flash memory, display drivers

---

## Why Should I Care? (STM32/Jetson Project Context)

SPI is the CORE protocol of your project. The STM32 Zephyr firmware reads the ICM-42688 IMU via SPI, and sends data to the Jetson Orin via another SPI bus. Understanding SPI at the shift-register level lets you:
- Debug the "first byte garbage" problem when the slave DMA isn't armed in time
- Choose the correct CPOL/CPHA mode for your IMU (Mode 3 for ICM-42688)
- Understand why 40MHz SPI on long jumper wires produces corrupted data
- Design the framing protocol (header + length + CRC) that rides on top of raw SPI bytes
- Know why SPI slave mode is fundamentally harder than master mode

---

# PART 1 — WHAT SPI IS: SHIFT REGISTERS IN A RING

---

## 1.1 The Core Insight — Two Shift Registers

Forget everything you think you know about SPI "sending" and "receiving." SPI is much simpler and more elegant:

**Master and Slave each have a shift register. They are connected in a ring. On every clock edge, they exchange one bit. After 8 clocks, they have exchanged one byte. That's it. That's all SPI is.**

```
    MASTER                                      SLAVE
    ┌──────────────────────┐     MOSI      ┌──────────────────────┐
    │ MSB → [7][6][5][4]   │──────────────→│ → [7][6][5][4]  MSB │
    │       [3][2][1][0]   │               │   [3][2][1][0]      │
    │               ↑  MSB │     MISO      │       ↑             │
    │               └──────│←──────────────│───────┘ shift out   │
    └──────────────────────┘               └──────────────────────┘
               │                                      │
              SCLK ──────────────────────────────────→│
               │                                      │
              CS̄  ──────────────────────────────────→│
```

**One clock cycle:**
1. Master shifts its MSB out onto MOSI wire
2. Slave shifts its MSB out onto MISO wire
3. Master shifts the MISO bit into its LSB position
4. Slave shifts the MOSI bit into its LSB position
5. Both shift registers advance by one position

After 8 clocks: **the master has the slave's original byte, and the slave has the master's original byte.** This is a physical EXCHANGE, not a "send" or "receive."

**ELI15 — the conveyor belt of sushi:**

Imagine a circular conveyor belt connecting two chefs. Each has 8 plates. On every tick of a timer:
- Each chef pushes their next plate onto the belt (from their end)
- Each chef receives the other's plate (arriving at their end)
- After 8 ticks, all 8 plates have been swapped

The belt runs continuously — you ALWAYS send AND receive simultaneously. If you only want to send, the received data is garbage (you ignore it). If you only want to receive, you must send something (usually 0x00 or 0xFF as "dummy bytes") to generate the clock that moves the slave's data out.

**This is why SPI is always full-duplex** — it's physically impossible for it NOT to be. The ring topology means every bit sent is simultaneously a bit received.

---

## 1.2 The 4 SPI Wires

| Wire | Full Name | Direction | Purpose |
|------|-----------|-----------|---------|
| SCLK | Serial Clock | Master → Slave | Timing reference for bit shifting |
| MOSI | Master Out, Slave In | Master → Slave | Data from master to slave |
| MISO | Master In, Slave Out | Slave → Master | Data from slave to master |
| CS̄ | Chip Select (active low) | Master → Slave | "Hey you, I'm talking to you" |

**Modern naming:** MOSI/MISO are being replaced by SDO/SDI (Serial Data Out/In) or COPI/CIPO (Controller Out/Peripheral In) in newer datasheets. The function is identical.

---

# PART 2 — THE 4 SPI MODES

---

## 2.1 CPOL and CPHA — What They Mean

SPI has two parameters that define when data is shifted and sampled:

- **CPOL (Clock Polarity):** What is the clock level when idle?
  - CPOL=0: Clock idles LOW
  - CPOL=1: Clock idles HIGH

- **CPHA (Clock Phase):** When is data sampled relative to clock edges?
  - CPHA=0: Data sampled on the FIRST clock transition (leading edge)
  - CPHA=1: Data sampled on the SECOND clock transition (trailing edge)

**The 4 combinations:**

| Mode | CPOL | CPHA | Clock Idle | Sample Edge | Shift Edge |
|------|------|------|------------|-------------|------------|
| 0 | 0 | 0 | LOW | Rising ↑ | Falling ↓ |
| 1 | 0 | 1 | LOW | Falling ↓ | Rising ↑ |
| 2 | 1 | 0 | HIGH | Falling ↓ | Rising ↑ |
| 3 | 1 | 1 | HIGH | Rising ↑ | Falling ↓ |

## 2.2 ASCII Timing Diagrams

### Mode 0 (CPOL=0, CPHA=0) — Most Common

Data is set up BEFORE the first clock edge. Sampled on rising edges.

```
    Transmitting 0xA5 = 1010_0101 (MSB first)

    CS̄:   ‾‾‾‾|_______________________________________________|‾‾‾‾
                                                                 
    SCLK:      _|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_
               idle  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑  idle
                    sample on rising edges
                                                                 
    MOSI:  ‾‾‾|1   |0   |1   |0   |0   |1   |0   |1   |‾‾‾
               D7    D6    D5    D4    D3    D2    D1    D0
               MSB                                      LSB
                                                                 
    MISO:  ‾‾‾|x   |x   |x   |x   |x   |x   |x   |x   |‾‾‾
              (slave data comes out simultaneously)
```

Data setup: MOSI changes on FALLING edge (or before first clock), sampled on RISING edge.

### Mode 3 (CPOL=1, CPHA=1) — Used by ICM-42688

Clock idles HIGH. Data sampled on rising edges (same as Mode 0 sampling, but clock polarity is inverted).

```
    CS̄:   ‾‾‾‾|_______________________________________________|‾‾‾‾
                                                                 
    SCLK:  ‾‾‾‾‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾‾‾‾
               idle  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑  idle
                    sample on rising edges
                    (same as Mode 0!)
                                                                 
    MOSI:  ‾‾‾‾|1   |0   |1   |0   |0   |1   |0   |1   |‾‾‾
               D7    D6    D5    D4    D3    D2    D1    D0
```

**Key insight:** Mode 0 and Mode 3 both sample on rising edges. The ONLY difference is the idle clock level. Many devices (including ICM-42688) support both Mode 0 and Mode 3.

### Mode 1 and Mode 2

```
    Mode 1 (CPOL=0, CPHA=1): Idle LOW, sample on FALLING edge
    Mode 2 (CPOL=1, CPHA=0): Idle HIGH, sample on FALLING edge
    
    These are less common. Mode 1 is used by some ADCs and DACs.
```

## 2.3 What Happens with the Wrong Mode

If master is Mode 0 and slave is Mode 1:

```
    Master sends 0xA5 = 1010_0101
    Slave samples on opposite edge → sees bits shifted by half a period
    Slave reads:  0x4B or 0x52 or something else entirely
    
    The byte is WRONG, typically by 1 bit shift or complemented bits
```

**Diagnosis:** If the byte received is "close but wrong" (e.g., 0x4A instead of 0xA5), it's almost certainly a CPOL/CPHA mismatch. Check the slave device datasheet for the correct mode.

---

# PART 3 — CHIP SELECT (CS̄)

---

## 3.1 Active-Low Convention

CS̄ is active-LOW: the slave is selected (listening) when CS̄ = 0V, and deselected (ignoring the bus) when CS̄ = 3.3V.

**Why active-low?** Historical: early TTL logic consumed less power when outputs were LOW. Also, open-drain outputs naturally idle HIGH (pull-up) and must be actively driven LOW — matching the active-low convention.

### 3.2 Pull-Up During Boot

**Why should I care?** During MCU boot, GPIO pins are in a high-impedance state. If CS̄ floats LOW, the slave thinks it's selected and starts responding to noise on SCLK → corrupt data, confused slave state.

```
    VCC ──┤ 10kΩ ├── CS̄ pin ── MCU GPIO
                     │
                     └── To slave CS̄ input

    Pull-up ensures CS̄ = HIGH (slave deselected) during boot.
    MCU drives it LOW when ready to communicate.
```

### 3.3 Multi-Slave SPI

**Option 1: Separate CS̄ per slave (most common)**

```
    SCLK ──────────→ Slave 1 SCLK ──→ Slave 2 SCLK ──→ Slave 3 SCLK
    MOSI ──────────→ Slave 1 MOSI ──→ Slave 2 MOSI ──→ Slave 3 MOSI
    MISO ←── Slave 1 MISO
         ←── Slave 2 MISO     (directly connected — tri-state when not selected)
         ←── Slave 3 MISO
    
    CS1̄  ──────────→ Slave 1 CS̄
    CS2̄  ──────────→ Slave 2 CS̄
    CS3̄  ──────────→ Slave 3 CS̄
```

Only one CS̄ is LOW at a time. Non-selected slaves tri-state (high-impedance) their MISO output so they don't conflict.

**Option 2: Daisy-chain**

Slaves are chained: MISO of Slave 1 → MOSI of Slave 2 → etc. All slaves share one CS̄. Data shifts through ALL slaves in the chain. Less common, used by LED drivers (WS2812's SPI cousin) and some DAC chains.

---

# PART 4 — SIGNAL INTEGRITY AND CLOCK SPEED

---

## 4.1 Why 40MHz SPI on 30cm Jumper Wires Fails

**The short version:** At 40MHz, the clock period is 25ns. At the speed of light in copper (~2/3 c), a signal travels about 20cm in 1ns. A 30cm wire introduces ~1.5ns of propagation delay — that's 6% of a bit period. Add reflections from impedance mismatches, and the signal at the slave looks nothing like what the master sent.

**The longer version:**

At low frequencies, wires behave like ideal conductors. At high frequencies, wires become **transmission lines** with characteristic impedance (~50-120Ω depending on geometry). If the line impedance doesn't match the source/load impedance, the signal partially reflects at the far end, bouncing back and forth and creating ringing:

```
    What master sends (clean):        What slave sees (on 30cm jumper):
    
    ‾‾‾|     |‾‾‾|     |‾‾‾          ‾‾╱╲  ╱╲|‾‾╱╲ ╱╲|‾‾‾
        |     |   |     |                ╲╱╲╱  |  ╲╱╲╱ |
        |_____|   |_____|                 |‾‾‾‾|   |‾‾|
                                          ↑
                                      Ringing! Crosses threshold
                                      multiple times → extra clock edges
                                      → corrupted data
```

### 4.2 Rules of Thumb for SPI Signal Integrity

| SPI Speed | Max Wire Length | Notes |
|-----------|----------------|-------|
| ≤1 MHz | 1 meter | Anything works (jumper wires, breadboard) |
| 1-10 MHz | 30cm | Short wires, keep GND close to SCLK |
| 10-20 MHz | 10cm | PCB traces preferred, ground plane |
| 20-50 MHz | 5cm | PCB only, impedance-matched traces, series termination |
| >50 MHz | <3cm | Tight PCB layout, matched length, microstrip calculations |

**For your 100Hz bridge at 10MHz SPI:** Keep wires under 15cm, ensure GND wire runs alongside SCLK, and you'll be fine.

### 4.3 Why a GND Wire Alongside SCLK Helps

The return current for the SCLK signal needs a path back. If the nearest GND wire is 10cm away, the return current loop is large → more inductance → more EMI radiation → more ringing.

Place a GND wire physically adjacent to (or twisted with) the SCLK wire. This minimizes the loop area and reduces ringing.

```
    GOOD:                           BAD:
    SCLK ════════════════          SCLK ════════════════
    GND  ════════════════          
    MOSI ════════════════          MOSI ════════════════
    GND  ════════════════          
    MISO ════════════════          MISO ════════════════
    CS̄   ════════════════          CS̄   ════════════════
                                   GND  ════════════════ (far away!)
    (signal-ground-signal          (all signals far from ground)
     sandwich → low loop area)
```

On a PCB, this is achieved with a continuous ground plane on the layer beneath the signal traces.

---

# PART 5 — SPI MASTER vs SLAVE TIMING

---

## 5.1 Why Master Mode is Easy

The master generates the clock. It knows exactly when each bit will be sampled. It can set up data on MOSI well before the clock edge, and sample MISO well after the clock edge. The master is always in control.

## 5.2 Why Slave Mode is HARD

The slave doesn't control the clock. It must:

1. **Detect CS̄ going LOW** (interrupt or DMA trigger)
2. **Arm the shift register with data to send** (load the first byte into the TX register)
3. **Be ready before the first SCLK edge arrives**

Step 3 is the problem. If the master asserts CS̄ and starts clocking within nanoseconds, the slave may not have armed its DMA in time.

```
    TIMELINE:

    CS̄ asserts ──→ Slave interrupt fires ──→ ISR runs ──→ DMA armed ──→ Ready!
    |              |                          |             |
    |  ~50ns       |  ~200ns (interrupt       |  ~500ns     |
    |  (wire       |   latency on CM7)        |  (DMA       |
    |  propagation |                          |  setup)     |
    |              |                          |             |
    ↓              ↓                          ↓             ↓
    Meanwhile, master starts clocking...
    |  1st SCLK edge at ~100ns after CS̄
    |
    PROBLEM: Slave DMA not armed yet!
    First byte sent by slave = whatever was left in the shift register = GARBAGE
```

### The "First Byte Garbage" Problem

This is the most common SPI slave bug. The slave's first byte is wrong because the shift register wasn't loaded with valid data before the master started clocking.

**Fixes:**
1. **Master waits:** Insert a delay (1-10µs) between CS̄ assertion and first clock edge. Simple but wastes time.
2. **Pre-arm the DMA:** Keep the slave DMA perpetually armed with a transmit buffer. When CS̄ asserts, data is already in the shift register.
3. **Use a protocol with a sychonization byte:** The master sends a "wake up" byte (0x00) first, which the slave echoes back as junk. Real data starts from byte 2.
4. **NSS hardware management:** STM32 SPI slave can detect CS̄ assertion in hardware and auto-load the shift register from the TX FIFO — faster than software interrupt.

---

## 5.3 SPI DMA — Essential for Slave at High Speed

Without DMA, the CPU must load each byte into the TX register and read each byte from the RX register. At 10MHz SPI, that's a new byte every 800ns. The CPU can barely keep up (especially with other interrupts).

With DMA: the DMA controller moves bytes between RAM and the SPI peripheral automatically. The CPU is only interrupted when the entire buffer is complete.

```c
/* Zephyr SPI slave with DMA (conceptual): */
struct spi_config slave_cfg = {
    .frequency = 10000000,  /* 10MHz */
    .operation = SPI_OP_MODE_SLAVE | SPI_WORD_SET(8) |
                 SPI_TRANSFER_MSB | SPI_MODE_CPOL | SPI_MODE_CPHA,
    .cs = NULL,  /* CS managed by hardware in slave mode */
};

uint8_t tx_buf[64] = { /* sensor data */ };
uint8_t rx_buf[64] = { 0 };

struct spi_buf tx = { .buf = tx_buf, .len = 64 };
struct spi_buf rx = { .buf = rx_buf, .len = 64 };
struct spi_buf_set tx_set = { .buffers = &tx, .count = 1 };
struct spi_buf_set rx_set = { .buffers = &rx, .count = 1 };

/* This blocks until CS̄ asserts and 64 bytes are exchanged */
spi_transceive(spi_dev, &slave_cfg, &tx_set, &rx_set);
```

---

## 5.4 STM32H7 SPI Specifics

The STM32H7 SPI peripheral has:
- **32-byte FIFO** for TX and RX
- **DMA support** with configurable threshold
- **Hardware NSS management** for slave mode
- **16/32-bit word width** for efficient data transfer
- **Kernel clock** up to 150MHz → max SPI clock ~75MHz (ƒ_pclk/2)

**D-Cache trap (critical for Cortex-M7!):**

The STM32H7 has an L1 data cache. DMA operates on physical memory, NOT through the cache. If the CPU writes to a buffer that's cached, the DMA reads stale data from physical RAM.

```
    CPU writes to tx_buf:  CPU cache has new data, RAM has OLD data
    DMA reads from RAM:    DMA sends OLD data on SPI ← BUG!

    Fix options:
    1. Place DMA buffers in non-cacheable memory region (MPU configuration)
    2. Clean (flush) the cache before DMA TX: SCB_CleanDCache_by_Addr(tx_buf, len)
    3. Invalidate the cache after DMA RX: SCB_InvalidateDCache_by_Addr(rx_buf, len)
    4. Use DTCM RAM for DMA buffers (not cached, directly connected to bus)
```

This is the #1 source of "SPI works in debug mode but fails at speed" on STM32H7.

---

# PART 6 — SPI PROTOCOL FRAMING

---

## 6.1 Why You Need a Protocol on Top of SPI

Raw SPI is just byte exchange. It has no concept of:
- Message boundaries (where does one message end and the next begin?)
- Error detection (was the byte corrupted?)
- Sequence tracking (did we miss a message?)
- Synchronization (if the slave starts mid-byte, everything is shifted)

You must build ALL of this in software.

## 6.2 Basic Framing Protocol

```
    ┌────────┬────────┬──────────────────────┬────────┐
    │ MAGIC  │ LENGTH │      PAYLOAD         │  CRC   │
    │ 0xAA55 │ 1 byte │   LENGTH bytes       │ 2 bytes│
    │ 2 bytes│        │                      │ CRC-16 │
    └────────┴────────┴──────────────────────┴────────┘
```

**Magic bytes (0xAA55):** The receiver scans for this pattern to find the start of a frame. Chosen because the bit pattern 10101010_01010101 is distinctive and unlikely to appear randomly in data.

**Length:** Number of bytes in the payload (0-255 for 1-byte length, or 0-65535 for 2-byte length).

**CRC:** Cyclic Redundancy Check over the entire frame (magic + length + payload). CRC-16-CCITT is common. If the CRC doesn't match, the frame is corrupted → discard it.

**Sequence numbers:** Add a 1-byte incrementing counter in the header. If the receiver sees seq=5, then seq=7, it knows frame 6 was dropped. Essential for your 100Hz bridge — the Jetson needs to know if it missed an IMU sample.

```
    Extended frame with sequence number:

    ┌────────┬────────┬───────┬──────────────────┬────────┐
    │ MAGIC  │  SEQ   │  LEN  │    PAYLOAD       │  CRC   │
    │ 0xAA55 │ 1 byte │1 byte │   LEN bytes      │ 2 bytes│
    └────────┴────────┴───────┴──────────────────┴────────┘
    
    Total overhead: 2 + 1 + 1 + 2 = 6 bytes per frame
    For 52-byte IMU packet: 6/58 = 10.3% overhead — acceptable
```

## 6.3 The Slave Re-Sync Problem

If the slave DMA de-synchronizes (e.g., missed a CS̄ edge, noise triggered extra clock), the slave may start reading mid-frame. Everything is shifted by N bytes → all subsequent frames are corrupted.

**Recovery protocol:**
1. Master detects CRC failures (slave sends bad data)
2. Master sends a "re-sync" command: 10× 0xFF followed by the magic bytes
3. Slave receives 0xFF stream, recognizes it as "reset," re-arms its DMA
4. Slave detects magic bytes in the next frame → synchronized again

---

# PART 7 — SPI vs I2C vs UART COMPARISON

---

| Property | SPI | I2C | UART |
|----------|-----|-----|------|
| **Wires** | 4+ (SCLK, MOSI, MISO, CS̄ per slave) | 2 (SDA, SCL) + pull-ups | 2 (TX, RX) + GND |
| **Speed** | ≤100MHz (typically ≤50MHz) | 100kHz – 3.4MHz | 9600 – 3Mbaud |
| **Duplex** | Full duplex (always) | Half duplex | Full duplex |
| **Addressing** | CS̄ pin per slave | 7-bit address in protocol | Point-to-point (no addressing) |
| **Max slaves** | Limited by CS̄ pins | 127 (7-bit address) | 1 (point-to-point) |
| **Clock** | Master provides | Master provides | No clock (asynchronous) |
| **Complexity** | Simple hardware, complex slave DMA | Complex protocol, simple hardware | Simple everything |
| **Distance** | <30cm at high speed | <1m with proper pull-ups | >10m (RS-485: 1200m) |
| **Error detection** | None built-in (add CRC in protocol) | ACK/NACK per byte | None built-in (add parity) |
| **Typical use** | IMU, flash, display, high-speed sensor | EEPROM, temp sensor, IO expander | Debug console, GPS, RS-485 |
| **CPU overhead** | Low (DMA-friendly) | Moderate (bit-level protocol) | Low (DMA-friendly) |

**When to use each:**
- **SPI:** When you need SPEED (>400kHz) or periodic polling at fixed rates (your 100Hz bridge)
- **I2C:** When you need many slow sensors on one bus with minimal wires
- **UART:** When you need point-to-point communication, especially over long distances (RS-485)

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| All bytes read as 0xFF | CS̄ not going LOW, or MISO pull-up keeping line high | Scope CS̄ during transaction. Check wiring. | Fix CS̄ wiring. GPIO configured as output? |
| Bytes are shifted by 1 bit | Wrong CPOL/CPHA mode | Scope MOSI/MISO + SCLK. Check which edge data changes on. | Match mode to device datasheet |
| First byte from slave is garbage | Slave DMA not armed before master clocks | Known issue — see Section 5.2 | Pre-arm DMA, or add CS̄-to-SCLK delay |
| Works at 1MHz, fails at 10MHz | Signal integrity — ringing on long wires | Scope SCLK at slave end — look for overshoot/ringing | Shorter wires, series termination (33Ω on SCLK), ground plane |
| Data correct sometimes, corrupt other times | D-Cache coherency (STM32H7) | Works in debug (slow), fails at speed | Place buffers in non-cacheable RAM, or SCB_CleanDCache/InvalidateDCache |
| Wrong byte order | MSB vs LSB setting mismatch | Compare first byte expected vs received. Are they bit-reversed? | Match SPI_TRANSFER_MSB / SPI_TRANSFER_LSB to device |
| Slave stops responding | CS̄ de-glitched (noise), slave thinks transaction ended | Scope CS̄ — look for brief HIGH glitches during transaction | Add 100pF cap on CS̄ line, reduce CS̄ trace length |
| MISO floating when slave not selected | Slave doesn't tri-state MISO when CS̄ = HIGH | Measure MISO voltage when no slave selected — should be indeterminate | Normal for single-slave. For multi-slave, check slave tri-state spec |
| DMA transfer size wrong | FIFO threshold misconfigured (STM32H7) | HAL SPI FIFO threshold must match transfer word size | Set FIFO threshold to match SPI data width |
| Slave receives all zeros | MOSI wire broken or not connected | Scope MOSI at slave — should see data during SCLK | Check physical connection |
| Intermittent CRC failures in protocol | EMI, ground noise, or slight timing margin | Increase CRC check. Retry once. Monitor error rate. | Reduce SPI speed, improve grounding, add decoupling |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── SPI CHEAT SHEET ────────────────────────────────────────────┐
│                                                                                              │
│  CORE CONCEPT: Two shift registers in a ring. 8 clocks = 1 byte exchanged. Always FULL DX. │
│                                                                                              │
│  WIRES:   SCLK (Master→Slave)   MOSI (M→S)   MISO (S→M)   CS̄ (M→S, active LOW)          │
│                                                                                              │
│  MODES:   Mode 0 (0,0)  idle LOW,  sample ↑     ← most common                              │
│           Mode 1 (0,1)  idle LOW,  sample ↓                                                 │
│           Mode 2 (1,0)  idle HIGH, sample ↓                                                 │
│           Mode 3 (1,1)  idle HIGH, sample ↑     ← ICM-42688 uses this                      │
│                                                                                              │
│  CLOCK RATES:                                                                                │
│    ≤1MHz:  breadboard OK            10-20MHz: short wires, GND nearby                       │
│    1-10MHz: jumpers <15cm           >20MHz: PCB only, impedance match                       │
│                                                                                              │
│  FRAMING:  [MAGIC 0xAA55] [SEQ] [LEN] [PAYLOAD] [CRC-16]                                  │
│                                                                                              │
│  STM32H7 TRAPS:                                                                             │
│    1. D-Cache vs DMA → use non-cacheable memory or flush/invalidate                         │
│    2. FIFO threshold must match word size                                                    │
│    3. Slave first-byte-garbage → pre-arm DMA or add CS̄ delay                               │
│                                                                                              │
│  SLAVE MODE CHECKLIST:                                                                       │
│    □ CS̄ pull-up during boot (10kΩ)                                                          │
│    □ DMA for both TX and RX                                                                  │
│    □ Pre-arm DMA before master starts                                                        │
│    □ Protocol with magic bytes for re-sync                                                   │
│    □ CRC for error detection                                                                 │
│    □ Sequence numbers for drop detection                                                     │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
