# Exercises: SPI — Serial Peripheral Interface

### Chapter 05: From shift registers to frames — the protocol YOUR bridge uses

**Self-assessment guide:** Write your answer before expanding the details block. SPI is the CORE protocol of your STM32H7 ↔ Jetson Orin bridge at 100Hz. Spend extra time on the shift register model and CPOL/CPHA — getting these wrong will cost you days of debugging.

**Project context:** The STM32 reads an ICM-42688 IMU via SPI (Mode 3), then sends data to the Jetson Orin via another SPI bus (slave mode). You need to understand the exchange model, timing modes, signal integrity, and the infamous "first byte garbage" problem.

---

## Section A — Conceptual Questions

**A1.** Explain the "shift register ring" model of SPI. Why is it physically impossible for SPI to be half-duplex? Use the conveyor belt analogy if it helps.

<details><summary>Answer</summary>

SPI consists of **two shift registers (one in master, one in slave) connected in a ring**:
- Master's MSB shifts out onto MOSI → enters slave's LSB
- Slave's MSB shifts out onto MISO → enters master's LSB
- After 8 clock cycles, the master and slave have **exchanged** their bytes

This is a physical ring topology — every bit pushed out of one end is simultaneously received at the other. There is no "send mode" or "receive mode." **Every clock cycle both sends AND receives one bit.** It's like a circular conveyor belt: pushing a plate onto your end automatically delivers a plate from the other end.

This means:
- **"Read" operations** still require sending something (dummy bytes, usually 0x00 or 0xFF) to generate the clocks that shift the slave's data out.
- **"Write" operations** still receive something (the slave's shift register contents, typically ignored).
- There is no way to "just listen" without also transmitting, because the ring must flow in both directions simultaneously.

This is fundamentally different from I2C (half-duplex, shared SDA line) or UART (independent TX/RX, but not physically linked).

</details>

---

**A2.** Your ICM-42688 IMU uses SPI Mode 3. What are the CPOL and CPHA values? What is the idle clock level, and on which edge is data sampled? Why do Mode 0 and Mode 3 both sample on the same edge type?

<details><summary>Answer</summary>

Mode 3: **CPOL=1, CPHA=1**

- **CPOL=1:** Clock idles **HIGH**
- **CPHA=1:** Data sampled on the **second (trailing) clock transition**
- Since CPOL=1, the clock goes HIGH→LOW→HIGH. The first transition is falling, the second is rising.
- Therefore, **data is sampled on the rising edge** in Mode 3.

Mode 0 (CPOL=0, CPHA=0) also samples on the **rising edge** — because:
- CPOL=0: Clock idles LOW. First transition is rising.
- CPHA=0: Sample on first (leading) edge = rising edge.

**Both Mode 0 and Mode 3 sample on rising edges.** The ONLY difference is the idle clock level. This is why many devices (including ICM-42688) support both Mode 0 and Mode 3 — the actual sampling behavior is identical; only the clock polarity between transactions differs.

</details>

---

**A3.** You connect your SPI master (Mode 0) to a slave that expects Mode 1. The master sends 0xA5. What kind of error will you see? Will the received byte be completely random, or will there be a pattern?

<details><summary>Answer</summary>

Mode 0 samples on **rising** edges. Mode 1 samples on **falling** edges. The slave is sampling at the **wrong clock edge** — exactly half a clock period off from where it should.

The error is **NOT random**. The slave samples each bit shifted by half a period, which typically results in:
- A **1-bit shift** of the entire byte (each bit is from the adjacent bit-time)
- The received value is systematically wrong — e.g., 0xA5 might be read as 0x4B or 0x52

**Diagnostic pattern:** If the byte received is "close but wrong" — consistently off by a bit rotation or complement — it's almost certainly a **CPOL/CPHA mismatch**. Truly random corruption would indicate a different problem (clock speed, wiring, noise).

**Fix:** Check the slave device datasheet for the correct SPI mode. On STM32 Zephyr, set `SPI_MODE_CPOL` and `SPI_MODE_CPHA` flags in the `spi_config.operation` field.

</details>

---

**A4.** Why does CS̄ (Chip Select) need a pull-up resistor to VCC? What happens during STM32 boot if CS̄ floats LOW?

<details><summary>Answer</summary>

During MCU boot (reset, bootloader, clock configuration), **GPIO pins are in high-impedance state** (not actively driven). If the CS̄ line has no pull-up:

- CS̄ floats at an indeterminate voltage — noise can pull it LOW
- The slave interprets LOW as "I'm selected" and starts responding to whatever noise appears on SCLK
- The slave's shift register gets loaded with garbage data
- The slave's internal state machine may enter a confused state

A **10kΩ pull-up to VCC** ensures CS̄ = HIGH (slave deselected) during boot. The MCU only drives it LOW when it has finished initialization and is ready to communicate.

This is especially important for **SPI flash** chips: if CS̄ floats LOW during boot, the flash may interpret noise as erase or write commands, corrupting stored firmware. Always pull up CS̄.

</details>

---

**A5.** Explain the "first byte garbage" problem in SPI slave mode. Why doesn't this problem exist in master mode? What are 3 strategies to fix it?

<details><summary>Answer</summary>

**The problem:** When the master asserts CS̄ and immediately starts clocking, the slave may not have armed its DMA/shift register in time. The sequence is:

1. CS̄ goes LOW (~0 ns)
2. Slave interrupt fires (~50 ns wire propagation + ~200 ns interrupt latency)
3. ISR runs and arms DMA (~500 ns)
4. Meanwhile, master's first SCLK edge arrives at ~100 ns after CS̄

The slave's shift register hasn't been loaded yet → the first byte sent by the slave is whatever stale data was left in the register = **garbage**.

**Why master mode is immune:** The master generates the clock. It sets up MOSI data first, then starts clocking. It's always in control of timing. The slave has no such luxury — it must react to an externally-generated clock.

**3 fixes:**
1. **Master waits:** Insert a 1–10 µs delay between CS̄ assertion and first SCLK edge. Simple but wastes time on every transaction.
2. **Pre-arm DMA:** Keep the slave DMA perpetually armed with a transmit buffer. When CS̄ asserts, data is already loaded in the shift register from FIFO/DMA. This is the recommended approach for your 100Hz bridge.
3. **Protocol sync byte:** The master sends a dummy byte first (0x00), which the slave echoes back as junk. Real data starts from byte 2. Both sides agree to discard the first byte.

STM32H7's **hardware NSS management** can auto-load the shift register from the TX FIFO on CS̄ assertion — faster than any software interrupt approach.

</details>

---

**A6.** What is the D-Cache coherency trap on STM32H7, and why is it the #1 source of "SPI works in debug mode but fails at speed"?

<details><summary>Answer</summary>

The Cortex-M7 in STM32H7 has an **L1 data cache** for performance. DMA operates on **physical RAM**, NOT through the cache. This creates a coherency problem:

**TX path bug:**
1. CPU writes sensor data to `tx_buf` → data lands in **cache**, not yet in RAM
2. DMA reads `tx_buf` from **RAM** → sends stale/old data on SPI

**RX path bug:**
1. DMA writes received data to `rx_buf` in **RAM**
2. CPU reads `rx_buf` → cache returns **old cached data**, not the DMA's new data

**Why it works in debug mode:** The debugger halts the CPU frequently, causing cache flushes. At full speed, the cache coherency gap becomes visible.

**Fixes (pick one):**
1. Place DMA buffers in **non-cacheable memory** (configure MPU)
2. `SCB_CleanDCache_by_Addr(tx_buf, len)` before DMA TX (flush cache → RAM)
3. `SCB_InvalidateDCache_by_Addr(rx_buf, len)` after DMA RX (discard cache → re-read from RAM)
4. Place buffers in **DTCM RAM** (tightly coupled, not cached, directly on the bus)

Option 4 (DTCM) is the cleanest for small, frequently-accessed DMA buffers. Option 1 (MPU) is best for large buffers.

</details>

---

**A7.** In a multi-slave SPI setup with 3 slaves sharing SCLK/MOSI/MISO, what happens if two CS̄ lines are accidentally driven LOW at the same time? What physical problem occurs on the MISO line?

<details><summary>Answer</summary>

Both slaves think they're selected and both drive their MISO output at the same time. This creates a **bus contention** on the MISO line:

- Slave 1 drives MISO = HIGH (3.3V through its push-pull output)
- Slave 2 drives MISO = LOW (0V through its push-pull output)
- Result: direct short circuit — VCC through the two output drivers to GND

This causes:
1. **Corrupted data:** The master reads an indeterminate voltage on MISO
2. **Excessive current draw:** Both output drivers fight each other, potentially drawing 20–100 mA through the GPIO pins
3. **Possible damage:** Sustained contention can overheat and destroy the output transistors in both slaves

In normal operation, non-selected slaves **tri-state** their MISO output (high-impedance), so only the selected slave drives the line. The single-CS̄-at-a-time rule must be enforced in software.

</details>

---

**A8.** Why must you add a protocol layer (header + length + CRC + sequence number) on top of raw SPI for your 100Hz bridge? What does raw SPI NOT provide that you need?

<details><summary>Answer</summary>

Raw SPI is just byte exchange — it has **no concept of:**

1. **Message boundaries:** SPI doesn't know where one IMU packet ends and the next begins. Without a header/magic bytes, the receiver can't find the start of a message.

2. **Error detection:** SPI has NO built-in error checking. A noise spike that flips a bit goes undetected. A CRC-16 catches corrupted frames.

3. **Synchronization:** If the slave starts mid-byte (DMA de-sync, missed CS̄ edge), everything is shifted by N bytes. Without magic bytes (e.g., 0xAA55) at the start of each frame, there's no way to re-synchronize.

4. **Drop detection:** At 100Hz, if the Jetson misses frame #47, it needs to know it jumped from #46 to #48. A **sequence number** makes drops visible; without it, the Jetson processes stale data as if it's fresh.

For your bridge, a minimal protocol is: `[0xAA55 magic][seq byte][length byte][52 bytes IMU payload][CRC-16]` = 57 bytes per frame, with 5 bytes (8.8%) overhead. This catches corruption, detects drops, and enables re-sync after faults.

</details>

---

**A9.** The STM32H7 SPI peripheral can run at up to ~75 MHz (kernel clock / 2). Your bridge only needs 10 MHz. Why not run at maximum speed for minimum latency?

<details><summary>Answer</summary>

Running faster than needed creates problems without meaningful benefit:

1. **Signal integrity:** At 75 MHz, the clock period is 13.3 ns. On jumper wires or a loose PCB layout, propagation delay (~1.5 ns per 30cm), ringing, and reflections cause the slave to see corrupted clock edges. At 10 MHz (100 ns period), there's 7× more timing margin.

2. **EMI radiation:** Higher SPI clock frequencies radiate more electromagnetic interference. At 75 MHz, harmonics enter the FM radio band. On a robot with sensitive IMU sensors, this EMI can couple into analog signals and degrade measurement accuracy.

3. **Diminishing returns:** At 100Hz with 57-byte frames, the SPI transfer takes 57 × 8 / 10,000,000 = **45.6 µs** at 10 MHz. At 75 MHz it would take 6.1 µs. The saving of 39.5 µs per frame (at 100Hz = 3.95 ms total per second) is negligible compared to the 10 ms frame period. You're bus-idle 99.5% of the time anyway.

4. **D-Cache and DMA setup overhead** dominate latency more than the actual SPI transfer time at these frame sizes. Faster SPI doesn't reduce setup time.

**Rule of thumb for SPI speed:** Use the minimum speed that comfortably fits your data within the frame period. Add 2–3× headroom for retransmissions, but no more.

</details>

---

## Section B — Spot the Bug

**B1.** An engineer configures their STM32 SPI master for Mode 0 (CPOL=0, CPHA=0) to communicate with an ICM-42688 IMU. The IMU datasheet says it supports Mode 0 and Mode 3. The engineer reads the WHO_AM_I register and gets the correct value (0x47). They declare the SPI link working. Later, accelerometer readings are occasionally off by ~1 LSB. What did they miss?

<details><summary>Answer</summary>

**This is a trick question — Mode 0 and Mode 3 both sample on rising edges, so the WHO_AM_I read IS correct and Mode 0 genuinely works with the ICM-42688.**

The ~1 LSB accelerometer error is **NOT a Mode bug**. It's normal sensor noise — 16-bit accelerometer readings have ±1 LSB quantization noise inherently.

The bug is in the engineer's **reasoning**: they were worried about CPOL/CPHA but the actual issue is elsewhere (or there is no issue). This exercise tests whether you can distinguish a real SPI mode mismatch (which produces obviously wrong bytes like 0x4B instead of 0x47) from normal sensor behavior.

**Lesson:** Don't over-diagnose. If the register reads are correct, the SPI mode is correct. ±1 LSB noise is expected from any ADC.

</details>

---

**B2.** A student connects an SPI slave device to their STM32 master using 40cm jumper wires. At 1 MHz, everything works perfectly. They increase the SPI clock to 20 MHz for faster throughput, and now every other byte is corrupted. The MOSI data from the master looks clean on the oscilloscope, but the SCLK signal at the slave end shows severe ringing with overshoot to 5V and undershoot to -1V. What's causing this and how do they fix it?

<details><summary>Answer</summary>

**Bug:** At 20 MHz, the clock period is 50 ns. The 40cm jumper wire introduces ~2 ns propagation delay, and the impedance mismatch between the MCU output (~30Ω source) and the uncontrolled wire (~50–120Ω characteristic impedance) causes **signal reflections** at the far end.

These reflections create ringing — the SCLK signal crosses the logic threshold multiple times per intended edge. The slave's input sees **extra clock edges** and shifts extra bits, corrupting the data.

The overshoot to 5V and undershoot to -1V also risks **exceeding the slave's absolute maximum ratings** (typically VDD + 0.3V and -0.3V), potentially damaging the input pin over time.

**Fixes (any combination):**
1. **Shorten the wires** to <10 cm (at 20 MHz, PCB traces are preferred)
2. **Add a 33Ω series resistor** on the SCLK line near the master output — this provides impedance matching and dampens reflections
3. **Run a GND wire alongside SCLK** to reduce the current loop area and inductance
4. **Reduce SPI speed** to 10 MHz, where the 100 ns period gives reflections time to settle
5. Move to a **PCB with ground plane** — the proper solution for >10 MHz SPI

</details>

---

**B3.** A developer configures their STM32H7 SPI slave with DMA. They place the TX and RX buffers in regular SRAM (not DTCM, not marked non-cacheable). The CPU fills `tx_buf` with IMU data, then calls `spi_transceive()`. The master reads the data and sees values from the **previous** frame — always one frame behind. What's the root cause?

<details><summary>Answer</summary>

**Bug:** This is the **D-Cache coherency** problem. The sequence is:

1. CPU writes new IMU data to `tx_buf` → data lands in **L1 D-Cache** but NOT in physical RAM
2. DMA reads `tx_buf` from physical **RAM** → sends the **old (stale) data** that was there from the previous frame
3. Eventually the cache line gets evicted and the new data reaches RAM — but by then, the DMA has already transmitted the old frame

This is why the data is always "one frame behind" — the DMA is reading the last frame's data from RAM while the current frame's data is trapped in cache.

**Fix (choose one):**
- Call `SCB_CleanDCache_by_Addr(tx_buf, sizeof(tx_buf))` **before** starting the DMA transfer — this flushes cache to RAM
- Place `tx_buf` in **DTCM RAM** (tightly coupled memory, not cached) using linker script attributes
- Configure the **MPU** to mark the buffer region as non-cacheable
- For RX: call `SCB_InvalidateDCache_by_Addr(rx_buf, sizeof(rx_buf))` **after** DMA completes

</details>

---

**B4.** An SPI slave is connected to a master with the following protocol: Master asserts CS̄, sends 4-byte command, reads 4-byte response. The master asserts CS̄ and immediately starts clocking without any delay. The slave's ISR arms the DMA on the CS̄ falling-edge interrupt. The first byte of every response is 0xFF regardless of what the slave loaded. Bytes 2–4 are correct. What's wrong?

<details><summary>Answer</summary>

**Bug:** Classic **"first byte garbage" problem.** The timeline:

1. CS̄ goes LOW → wire propagates to slave (~50 ns)
2. Slave GPIO interrupt fires (~200 ns after CS̄)
3. ISR enters and configures DMA (~300–500 ns total from CS̄)
4. Meanwhile, master starts SCLK ~100 ns after CS̄
5. First 8 clocks complete at ~800 ns (at 10 MHz) — **before the slave ISR has finished arming DMA**

The slave's shift register is un-loaded (power-on default = 0xFF) for byte 1. By byte 2, the DMA has armed and loaded valid data.

**Fixes:**
1. **Master adds delay:** Insert 2–5 µs between CS̄ assertion and first SCLK edge (simplest)
2. **Pre-arm DMA:** Keep slave DMA perpetually armed with the TX buffer. On CS̄, data is already in the shift register. Re-arm after each transaction completes.
3. **Protocol agreement:** Both sides agree byte 1 is a "sync byte" (discarded). Real data starts at byte 2.
4. **STM32H7 hardware NSS:** Enable hardware CS̄ management — the SPI peripheral auto-loads the shift register from FIFO before SCLK starts.

</details>

---

**B5.** A developer is debugging an SPI slave re-synchronization issue. The master detects 3 consecutive CRC failures and sends a re-sync sequence: the 2-byte magic (0xAA55) immediately followed by a new data frame. The slave never re-synchronizes. What's missing from the re-sync protocol?

<details><summary>Answer</summary>

**Bug:** The re-sync sequence doesn't **flush the slave's corrupted state** before sending the new magic bytes.

When the slave is de-synchronized, its DMA is mid-transfer — it's at some arbitrary byte offset within a frame. Sending 0xAA55 immediately means the slave may receive the magic bytes at an arbitrary alignment within its current DMA buffer. The 0xAA could land at offset 23 of a 57-byte buffer and be treated as payload, not a sync marker.

**What's missing:** The re-sync protocol should send a **flush sequence** first — for example, 10× 0xFF bytes — to fill any remaining buffer space with a known pattern. This ensures the slave's current DMA transfer completes (or times out). THEN send the magic bytes at the start of a new transfer.

**Proper re-sync sequence:**
1. Master de-asserts CS̄ (end current transaction)
2. Wait for slave DMA to timeout/reset
3. Master asserts CS̄ again
4. Send 10× 0xFF (flush)
5. Send 0xAA55 + new frame
6. Slave scans for 0xAA55 in received data to re-establish frame alignment

</details>

---

**B6.** An engineer measures the SPI clock frequency with an oscilloscope and sees 10 MHz as expected. But they calculate the data throughput as 10 Mbytes/sec. Their actual measured throughput is only 800 KB/sec for 64-byte frames at 100 Hz. Where's the gap in their reasoning?

<details><summary>Answer</summary>

**Bug:** Confusing **bus bit rate** with **effective data throughput.** Multiple factors reduce throughput:

1. **10 MHz SPI = 10 Mbit/sec = 1.25 MB/sec maximum** (not 10 MB/sec — they confused bits with bytes)

2. **Duty cycle:** The SPI bus is only active during CS̄ LOW. At 100 Hz with 64-byte frames, each transfer takes 64 × 8 / 10,000,000 = 51.2 µs. But the frame period is 10,000 µs. Bus utilization = 51.2 / 10,000 = **0.5%**. The bus is idle 99.5% of the time.

3. **Actual throughput:** 64 bytes × 100 Hz = **6,400 bytes/sec = 6.25 KB/sec** — not 800 KB/sec. If they're measuring 800 KB/sec, that's a different test.

The lesson: throughput depends on **how often you use the bus**, not just how fast it runs. For periodic sensor polling, the frame rate (100Hz) and frame size (64B), not the SPI clock speed, determine throughput.

</details>

---

## Section C — Fill in the Blank / From Memory

### C1: SPI Mode Table

Fill in the complete SPI mode table:

| Mode | CPOL | CPHA | Clock Idle | Sample Edge | Shift Edge | Example Device |
|------|------|------|-----------|-------------|------------|----------------|
| 0 | ___ | ___ | ___ | ___ | ___ | ___ |
| 1 | ___ | ___ | ___ | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ | ___ | ___ | ___ |

<details><summary>Answer</summary>

| Mode | CPOL | CPHA | Clock Idle | Sample Edge | Shift Edge | Example Device |
|------|------|------|-----------|-------------|------------|----------------|
| 0 | 0 | 0 | **LOW** | **Rising ↑** | **Falling ↓** | **Most common default, many flash chips** |
| 1 | 0 | 1 | **LOW** | **Falling ↓** | **Rising ↑** | **Some ADCs and DACs** |
| 2 | 1 | 0 | **HIGH** | **Falling ↓** | **Rising ↑** | **Less common** |
| 3 | 1 | 1 | **HIGH** | **Rising ↑** | **Falling ↓** | **ICM-42688 IMU (also supports Mode 0)** |

**Key memory aid:** Mode 0 and Mode 3 both sample on **rising edges** — they only differ in idle clock level. Mode 1 and Mode 2 both sample on **falling edges.**

</details>

---

### C2: SPI Wire Functions

Fill in the 4 SPI wires:

| Wire | Full Name | Direction | Purpose |
|------|-----------|-----------|---------|
| SCLK | __________ | __________ | __________ |
| MOSI | __________ | __________ | __________ |
| MISO | __________ | __________ | __________ |
| CS̄   | __________ | __________ | __________ |

<details><summary>Answer</summary>

| Wire | Full Name | Direction | Purpose |
|------|-----------|-----------|---------|
| SCLK | **Serial Clock** | **Master → Slave** | **Timing reference for bit shifting** |
| MOSI | **Master Out, Slave In** | **Master → Slave** | **Data from master to slave** |
| MISO | **Master In, Slave Out** | **Slave → Master** | **Data from slave to master** |
| CS̄   | **Chip Select (active LOW)** | **Master → Slave** | **Selects which slave to talk to** |

Modern naming: MOSI/MISO → SDO/SDI or COPI/CIPO (Controller Out Peripheral In / Controller In Peripheral Out). Function is identical.

</details>

---

### C3: SPI Speed vs Wire Length

Fill in the maximum recommended wire length:

| SPI Speed | Max Wire Length | Notes |
|-----------|----------------|-------|
| ≤ 1 MHz | __________ | __________ |
| 1 – 10 MHz | __________ | __________ |
| 10 – 20 MHz | __________ | __________ |
| 20 – 50 MHz | __________ | __________ |
| > 50 MHz | __________ | __________ |

<details><summary>Answer</summary>

| SPI Speed | Max Wire Length | Notes |
|-----------|----------------|-------|
| ≤ 1 MHz | **1 meter** | **Anything works (jumper wires, breadboard)** |
| 1 – 10 MHz | **30 cm** | **Short wires, keep GND close to SCLK** |
| 10 – 20 MHz | **10 cm** | **PCB traces preferred, ground plane** |
| 20 – 50 MHz | **5 cm** | **PCB only, impedance-matched traces, series termination** |
| > 50 MHz | **< 3 cm** | **Tight PCB layout, matched length, microstrip calculations** |

For your 100Hz bridge at 10 MHz: keep wires under 15 cm, run a GND wire alongside SCLK, and you'll be fine.

</details>

---

### C4: SPI vs I2C vs UART

Fill in the comparison:

| Property | SPI | I2C | UART |
|----------|-----|-----|------|
| Wire count | ___ | ___ | ___ |
| Max speed | ___ | ___ | ___ |
| Duplex | ___ | ___ | ___ |
| Addressing method | ___ | ___ | ___ |
| Built-in error detection | ___ | ___ | ___ |
| Best for | ___ | ___ | ___ |

<details><summary>Answer</summary>

| Property | SPI | I2C | UART |
|----------|-----|-----|------|
| Wire count | **4+ (SCLK, MOSI, MISO, CS̄/slave)** | **2 (SDA, SCL) + pull-ups** | **2 (TX, RX) + GND** |
| Max speed | **≤100 MHz (typically ≤50 MHz)** | **100 kHz – 3.4 MHz** | **9600 – 3 Mbaud** |
| Duplex | **Full duplex (always)** | **Half duplex** | **Full duplex** |
| Addressing method | **CS̄ pin per slave** | **7-bit address in protocol** | **Point-to-point (none)** |
| Built-in error detection | **None (add CRC)** | **ACK/NACK per byte** | **None (add parity/CRC)** |
| Best for | **High-speed periodic: IMU, flash, display** | **Many slow sensors, EEPROM, IO expanders** | **Debug console, GPS, RS-485 multi-drop** |

</details>

---

## Section D — Lab / Calculation Tasks

**D1.** You need to read a 14-byte IMU packet (6 bytes accel + 6 bytes gyro + 2 bytes temperature) from the ICM-42688 at 100 Hz via SPI Mode 3 at 8 MHz. The read transaction requires: 1 byte register address (with read bit set) + 14 bytes read = 15 bytes total (1 command + 14 dummy bytes out, 1 junk + 14 data bytes in). Calculate: (a) time per SPI transfer, (b) bus utilization at 100 Hz, (c) total bits exchanged per second.

<details><summary>Answer</summary>

**(a) Time per transfer:**
- Total bytes = 15 (1 address + 14 data/dummy)
- Total bits = 15 × 8 = 120 bits
- Time = 120 / 8,000,000 = **15 µs per transfer**
- Plus CS̄ setup/hold time (~100 ns each) = 15.2 µs total

**(b) Bus utilization at 100 Hz:**
- Frame period = 1/100 = 10,000 µs
- Transfer time = 15.2 µs
- Utilization = 15.2 / 10,000 = **0.152%**
- The bus is idle 99.85% of the time — you could run 10× more sensors on this SPI bus if needed.

**(c) Total bits per second:**
- 120 bits × 100 Hz = **12,000 bits/sec** = 1,500 bytes/sec
- This is a tiny fraction of the 8 Mbit/sec capacity. SPI is vastly over-provisioned for 100 Hz IMU reads, which is exactly what you want for timing margin.

</details>

---

**D2.** Your bridge protocol uses: [0xAA55 magic (2B)] [SEQ (1B)] [LEN (1B)] [PAYLOAD (52B)] [CRC-16 (2B)] = 58 bytes per frame. At 10 MHz SPI, calculate: (a) transfer time per frame, (b) maximum possible frame rate, (c) protocol overhead percentage.

<details><summary>Answer</summary>

**(a) Transfer time:**
- 58 bytes × 8 bits = 464 bits
- Time = 464 / 10,000,000 = **46.4 µs per frame**

**(b) Maximum frame rate:**
- Assuming negligible CS̄ inter-frame gap (~1–5 µs): usable time per frame ≈ 50 µs
- Max rate = 1,000,000 / 50 = **~20,000 frames/sec**
- At your target of 100 Hz, you're using 0.5% of the maximum capacity — enormous headroom for retransmissions or additional data.

**(c) Protocol overhead:**
- Overhead bytes: 2 (magic) + 1 (seq) + 1 (len) + 2 (CRC) = 6 bytes
- Useful payload: 52 bytes
- Total: 58 bytes
- Overhead = 6/58 = **10.3%**
- This is acceptable — the overhead buys you synchronization (magic), drop detection (seq), length validation (len), and corruption detection (CRC).

</details>

---

**D3.** Draw the SCLK and MOSI waveform for transmitting byte 0xC3 (binary 1100_0011) in SPI Mode 0 (CPOL=0, CPHA=0), MSB first. Label which edges are sample edges and which are shift edges.

<details><summary>Answer</summary>

0xC3 = 1100_0011 (MSB first: D7=1, D6=1, D5=0, D4=0, D3=0, D2=0, D1=1, D0=1)

Mode 0: Clock idles LOW. Data sampled on **rising** edges. Data shifts (changes) on **falling** edges (or before first clock).

```
    CS̄:   ‾‾‾|______________________________________________|‾‾‾‾
    
    SCLK:     _|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_
              idle ↑S  ↑S  ↑S  ↑S  ↑S  ↑S  ↑S  ↑S  idle
                   ↓H  ↓H  ↓H  ↓H  ↓H  ↓H  ↓H  ↓H
              S = Sample (rising)    H = sHift (falling)
    
    MOSI: ‾‾‾|1   |1   |0   |0   |0   |0   |1   |1   |‾‾‾
              D7    D6    D5    D4    D3    D2    D1    D0
              MSB                                     LSB
    
    Data is set up BEFORE or ON falling edge.
    Data is sampled ON rising edge.
    
    The receiver reads: 1,1,0,0,0,0,1,1 = 0xC3 ✓
```

**Key point:** In Mode 0, data on MOSI must be stable **before** the first rising edge. The master sets up D7=1 when CS̄ goes LOW (or slightly before the first SCLK rise).

</details>

---

**D4.** You have 3 SPI slave devices: an ICM-42688 IMU (Mode 3), an SPI flash (Mode 0), and a DAC (Mode 1). You have one SPI peripheral on the STM32. Can you communicate with all three on the same SPI bus? What must you do between transactions?

<details><summary>Answer</summary>

**Yes, you can share the bus** — SCLK, MOSI, and MISO are shared; each device has its own CS̄ line. Only one CS̄ is asserted at a time, so devices don't conflict on MISO.

**What you must do between transactions:**

The SPI peripheral must be **reconfigured** for the correct mode before each transaction:

1. **Before IMU transaction:** Set CPOL=1, CPHA=1 (Mode 3). Assert CS_IMU.
2. **Before flash transaction:** Set CPOL=0, CPHA=0 (Mode 0). Assert CS_FLASH.
3. **Before DAC transaction:** Set CPOL=0, CPHA=1 (Mode 1). Assert CS_DAC.

On STM32, this means updating the SPI_CR1 register's CPOL and CPHA bits, which requires:
1. Disable SPI peripheral (SPI_CR1.SPE = 0)
2. Change CPOL/CPHA bits
3. Re-enable SPI peripheral (SPI_CR1.SPE = 1)

In Zephyr, each device has its own `struct spi_config` with the correct mode flags. The driver handles the reconfiguration automatically when you call `spi_transceive()` with a different config.

**Caveat:** Mode switching adds ~1–2 µs overhead per transaction. If you're doing thousands of transactions per second across different modes, consider using separate SPI peripherals (STM32H7 has up to 6 SPI instances).

</details>

---

**D5.** The STM32H7 SPI peripheral has a 32-byte FIFO and maximum kernel clock of 150 MHz. What is the maximum SPI clock speed achievable? If you're running SPI slave mode at 10 MHz and the master sends 64-byte frames, how many bytes does the FIFO buffer buy you in terms of ISR response time?

<details><summary>Answer</summary>

**Maximum SPI clock:**
- SPI clock = kernel clock / prescaler
- Minimum prescaler = 2
- Max SPI clock = 150 MHz / 2 = **75 MHz**

**FIFO buffer time at 10 MHz slave:**
- At 10 MHz SPI, one byte = 8 bits / 10 MHz = **0.8 µs** per byte
- 32-byte FIFO = 32 × 0.8 = **25.6 µs of buffering**

This means: when the master starts a 64-byte transaction, the slave DMA or ISR has **25.6 µs** to start draining the RX FIFO before it overflows. Without the FIFO (direct shift-register-to-CPU), the ISR would need to respond within **0.8 µs** per byte — extremely tight.

For a 64-byte frame at 10 MHz:
- Total transfer time = 64 × 0.8 = **51.2 µs**
- FIFO holds first 32 bytes (25.6 µs) while DMA initializes
- DMA must start draining before byte 33 arrives, or FIFO overflows

This is why **pre-armed DMA** is essential for SPI slave at >1 MHz — even with the FIFO, the timing window is measured in microseconds.

</details>

---

## Section E — Deeper Thinking

**E1.** Your 100Hz STM32→Jetson SPI bridge works perfectly for weeks, then suddenly every 8th frame has a CRC error. The error rate is exactly 12.5% (1 in 8). The errors started after a firmware update that added a 50 µs ADC sampling routine running at 800 Hz on a timer interrupt. Diagnose the likely root cause and propose a fix. Think about DMA, interrupts, and the shift register.

<details><summary>Answer</summary>

**Diagnosis:** The ADC timer interrupt fires at 800 Hz. Your SPI frames arrive at 100 Hz. The ratio is 800/100 = 8 — an ADC interrupt hits during every 8th SPI frame.

The ADC ISR takes 50 µs. During SPI slave DMA, if the DMA needs to respond (e.g., half-transfer complete, or FIFO threshold) and the ADC ISR has higher priority (or the same priority and happens to preempt), the DMA service is **delayed by 50 µs**.

At 10 MHz SPI, the 32-byte FIFO overflows in 25.6 µs. If the DMA is blocked for 50 µs, the FIFO overflows → bytes lost → CRC mismatch.

The exact 1-in-8 pattern occurs because the 800 Hz ADC and 100 Hz SPI are phase-aligned due to deriving from the same system clock. Every 8th SPI frame, the ADC ISR lands squarely during the SPI transfer.

**Fixes:**
1. **Priority inversion:** Set SPI DMA interrupt priority higher than the ADC timer interrupt. The SPI DMA ISR runs briefly (just rearms the buffer pointer), so it won't delay the ADC significantly.
2. **Move ADC to DMA:** If the ADC sampling also uses DMA, it doesn't need a 50 µs ISR at all — the DMA handles both ADC and SPI without CPU contention.
3. **Offset the ADC timer:** Add a 2 ms phase offset to the ADC timer, spreading the collision points across different SPI frames. This reduces the problem but doesn't eliminate it.
4. **Use DTCM for SPI buffers:** Ensures DMA accesses aren't delayed by cache operations that might coincide with the ADC ISR.

**Root lesson:** In real-time systems, ISR timing interactions between unrelated peripherals are a leading cause of intermittent faults. Always analyze the worst-case interrupt timeline.

</details>

---

**E2.** Compare the SPI shift register exchange mechanism with I2C's open-drain wired-AND and CAN's differential dominant/recessive. All three solve the "multiple devices on wires" problem differently. What are the fundamental trade-offs in terms of speed, wire count, bus contention safety, and distance? Why is SPI the right choice for your high-speed, short-distance, two-device bridge?

<details><summary>Answer</summary>

| Property | SPI (Shift Ring) | I2C (Open-Drain Wired-AND) | CAN (Differential Dom/Rec) |
|----------|-----------------|---------------------------|---------------------------|
| **Topology** | Point-to-point ring per CS̄ | Shared bus, multi-master safe | Shared bus, multi-master safe |
| **Speed** | Up to ~100 MHz | Up to 3.4 MHz | Up to 8 Mbps (CAN FD) |
| **Wires** | 4 + 1 per slave | 2 total | 2 total (differential) |
| **Distance** | <30 cm at speed | <1 m | Up to 1 km |
| **Contention** | Impossible (separate CS̄) | Safe (wired-AND, 0 wins) | Safe (dominant wins) |
| **CPU load** | Low (DMA-friendly, no protocol) | Moderate (bit-level protocol, ACK handling) | Low (hardware handles arbitration, retransmission) |

**Why SPI for your bridge:**

1. **Speed:** You need 100 Hz × 58 bytes = 5,800 bytes/sec minimum. SPI at 10 MHz provides 1.25 MB/sec — 215× headroom. I2C at 400 kHz gives only ~50 KB/sec, and CAN classic only 1 Mbit (limited to 8 bytes/frame).

2. **Two devices only:** SPI's "1 CS̄ per slave" downside is irrelevant — you only have one slave (the Jetson). I2C's addressing and CAN's arbitration add complexity you don't need.

3. **Full duplex:** SPI's ring topology means the STM32 can send IMU data TO the Jetson and receive commands FROM the Jetson in the same transaction. I2C is half-duplex; CAN requires separate frames.

4. **Short distance:** The STM32 and Jetson are on the same board or within 10 cm — well within SPI's signal integrity limits.

5. **DMA friendliness:** SPI's simple shift register mechanism maps directly to fixed-length DMA transfers with zero per-byte CPU overhead. I2C's per-byte ACK/NACK requires more complex state machine handling.

The trade-off: SPI has no built-in error detection, addressing, or bus arbitration. For a two-device, short-distance, high-speed link — that's an acceptable trade, handled by your framing protocol (magic + seq + CRC).

</details>

---

**E3.** Design a diagnostic procedure for your 100Hz SPI bridge that a field engineer (non-firmware-expert) could follow. They have an oscilloscope with 4 channels and a logic analyzer. The symptom is "Jetson reports 30% packet drop rate." Create a step-by-step diagnostic flowchart with specific signals to probe and pass/fail criteria at each step.

<details><summary>Answer</summary>

**SPI Bridge Diagnostic Procedure — 30% Packet Drop**

**Equipment:** 4-channel oscilloscope or logic analyzer. Probes on: CH1=SCLK, CH2=MOSI, CH3=MISO, CH4=CS̄.

**Step 1: Verify CS̄ timing**
- Trigger on CS̄ falling edge
- **PASS:** CS̄ goes LOW at 100 Hz (10 ms period ± 1%). CS̄ LOW duration matches expected transfer time (~50 µs for 58 bytes at 10 MHz).
- **FAIL (no pulses):** Master SPI not configured or GPIO not assigned. Check firmware configuration.
- **FAIL (irregular timing):** Master scheduler issue. Check timer/thread priorities.

**Step 2: Verify SCLK quality at slave end**
- Probe SCLK at the Jetson's SPI input pin (not at the STM32 output)
- **PASS:** Clean square wave at 10 MHz, no ringing, amplitude 0V–3.3V, no glitches.
- **FAIL (ringing/overshoot):** Signal integrity issue. Shorten wires, add 33Ω series resistor on SCLK, add GND wire alongside SCLK.

**Step 3: Count bytes per transaction**
- Use logic analyzer to decode SPI at correct Mode (3) and speed (10 MHz)
- **PASS:** 58 bytes per CS̄ assertion. First 2 bytes = 0xAA55 (magic header).
- **FAIL (wrong byte count):** DMA misconfigured. Check buffer sizes.
- **FAIL (no magic bytes):** Firmware not sending protocol frames. Check transmit buffer.

**Step 4: Verify MISO data (slave response)**
- Check that MISO has valid data (not all 0xFF or all 0x00)
- **PASS:** Variable data on MISO, matching protocol format.
- **FAIL (all 0xFF):** Slave DMA not armed — first byte garbage problem. Check slave pre-arm.
- **FAIL (all 0x00):** MISO wire broken or slave not powered.

**Step 5: Check for periodic failures**
- Capture 100 consecutive frames. Count CRC failures on the Jetson side.
- **PASS:** 0% CRC failure → drops are in software (Jetson not reading fast enough).
- **FAIL (regular pattern, e.g., every Nth frame):** ISR contention with timer interrupt. Check interrupt priorities.
- **FAIL (random CRC failures):** Noise/EMI. Improve grounding, reduce wire length.

**Step 6: Software-side check (Jetson)**
- Verify the Jetson SPI driver is configured for the correct mode, speed, and frame size
- Check that the read thread runs at ≥100 Hz and doesn't block
- Monitor the sequence number field: if seq increments by 1 each frame, the STM32 side is healthy and the drops are on the Jetson receive side

This flowchart isolates hardware vs firmware vs software issues in 6 steps, each taking <5 minutes with proper probes.

</details>

---

**E4.** The shift register exchange model means that if you want to READ 14 bytes from an IMU, you must also SEND 14 bytes (dummy bytes) to generate the clocks. A student proposes: "This is wasteful — half the data on the wire is useless dummy bytes. Can we modify SPI to be half-duplex and save bandwidth?" Explain why this is misguided, both physically and practically.

<details><summary>Answer</summary>

**Why the student's thinking is wrong:**

**Physically:** The shift register ring IS the mechanism. Removing one direction doesn't save any hardware or time — the clock still needs to tick 8 times per byte, and the shift registers still advance. The "dummy bytes" don't cost anything: the MOSI wire carries 0x00 while MISO carries real data, but the clock runs at the same speed regardless. You cannot make the clock tick faster by removing MOSI.

**Bandwidth isn't wasted:** The bus bandwidth is determined by SCLK frequency × 1 bit per clock. Whether useful data or zeros ride on MOSI during a read doesn't change the transfer time. The dummy bytes are free — they require zero additional bus time.

**Practically, full-duplex is a feature:**
- In your bridge protocol, the STM32 sends IMU data on MOSI while simultaneously receiving commands from the Jetson on MISO — **both directions carry useful data in every transaction**
- Half-duplex SPI would actually require **twice the transactions** (one to send, one to receive), doubling bus time and latency

**What "half-duplex SPI" actually looks like:** Some protocols use a single bidirectional data line (3-wire SPI), where SDA switches direction. This saves 1 wire but adds complexity, a direction-switching turnaround delay, and makes DMA harder. It's used in space-constrained sensor interfaces, not for high-speed data bridges.

**The correct optimization mindset:** Don't try to save bandwidth that isn't constrained. At 100Hz × 58B, you use 0.5% of the 10 MHz SPI capacity. The "wasted" dummy bytes consume 0.25% of an already-idle bus. Focus engineering effort on real bottlenecks: DMA setup time, D-cache coherency, and interrupt latency.

</details>
