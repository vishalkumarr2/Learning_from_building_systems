# Exercises: UART & Serial Communication

### Chapter 04: From electrons to bytes — the simplest serial protocol

**Self-assessment guide:** Write your answer before expanding the details block. If you can answer 80% correctly without peeking, you're ready to move on. If not, re-read the relevant section and try again.

**Project context:** You're building an STM32H7 ↔ Jetson Orin SPI bridge at 100Hz. UART is used for your debug console (115200 baud via ST-Link), and understanding UART DMA patterns transfers directly to SPI DMA. Many warehouse robot sensors (LIDAR, GPS) also speak UART or RS-485.

---

## Section A — Conceptual Questions

**A1.** Why did serial communication "win" over parallel for almost everything except memory buses and internal chip interconnects? Name two specific physical problems with parallel at high speed.

<details><summary>Answer</summary>

Serial won because:

1. **Skew:** In parallel buses, each wire has slightly different propagation delay. At high frequencies, bits arrive at different times across the 8+ wires, limiting maximum clock rate to ~100 MHz for short distances.
2. **Wire count:** Parallel requires 8 data + clock + ground = 10+ wires, which is expensive, bulky, and creates more EMI.

Modern serial protocols achieve higher throughput than parallel ever could (USB 3: 5 Gbps, PCIe: 16 Gbps per lane) by running a single lane extremely fast. The absence of inter-lane skew is what allows this.

</details>

---

**A2.** UART is "asynchronous." What does this mean physically? What happens if the transmitter and receiver disagree on timing by 5%?

<details><summary>Answer</summary>

"Asynchronous" means there is **no shared clock wire** between transmitter and receiver. Both sides must independently agree on the baud rate (bit timing) before communication starts.

The receiver synchronizes on the **falling edge of each start bit**, then samples each subsequent bit in the middle of its expected bit-time. If clocks disagree by 5%:

- Per-byte drift = 10 bits × 0.05 = 0.5 bit-times
- At 0.5 bit-times of drift, the sampling point is right at the edge of the adjacent bit — this is the **corruption threshold**
- Some bytes will be read correctly, others will be corrupted, producing intermittent garbage

The practical rule: keep baud rate mismatch below **±2%** for reliable 8N1 communication.

</details>

---

**A3.** Your STM32 debug console is running at 115200 baud, 8N1. How many microseconds does it take to transmit one byte? How many bytes per second can you actually push through?

<details><summary>Answer</summary>

- Bit time = 1 / 115200 = **8.68 µs**
- One 8N1 frame = 10 bit-times (1 start + 8 data + 1 stop) = 10 × 8.68 = **86.8 µs per byte**
- Throughput = 115200 / 10 = **11,520 bytes/sec**

The 2 overhead bits (start + stop) cost you **20% of the raw baud rate**. This is why "throughput ≠ baud rate."

</details>

---

**A4.** You accidentally connect an RS-232 device (+12V / -12V levels) directly to the UART RX pin of your 3.3V STM32. What happens and why?

<details><summary>Answer</summary>

The STM32 GPIO input pin is rated for a maximum of **VDD + 0.3V = 3.6V** (some pins are 5V-tolerant, but not all). An RS-232 signal swings to **+12V** (logic 0) and **-12V** (logic 1).

- **+12V exceeds the absolute maximum rating** → current flows through the input's clamping diodes to VDD, potentially damaging the ESD protection, cooking the I/O cell, or destroying the entire GPIO port.
- **-12V drives the pin below GND** → current flows through the substrate diode to GND, potentially latching up the chip or destroying it.

**Fix:** Use a **MAX3232** level shifter between RS-232 and the MCU. It uses a charge pump to convert between ±12V and 0/3.3V safely.

</details>

---

**A5.** Why does UART transmit LSB first, and what practical problem does this create when bridging UART to SPI?

<details><summary>Answer</summary>

UART transmits **least significant bit first** (D0 before D7). The historical reason is that the receiver can start evaluating partial results before all bits arrive (e.g., "is this number odd?" after just 1 bit).

SPI typically transmits **MSB first** (configurable, but MSB-first is the default and most common). When bridging UART data to SPI (or vice versa), this **bit order mismatch** can produce mystifying byte value swaps. For example, 0xC1 (binary 1100_0001) transmitted LSB-first on UART appears as 0x83 (binary 1000_0011) if the SPI receiver interprets the same bits as MSB-first.

**Fix:** Be aware of bit order at each boundary and use the SPI peripheral's bit-order configuration (`SPI_TRANSFER_MSB` vs `SPI_TRANSFER_LSB` in Zephyr) to match.

</details>

---

**A6.** Explain the purpose of the UART idle-line interrupt in the context of DMA. Why can't you just wait for the DMA buffer to fill up?

<details><summary>Answer</summary>

DMA transfers bytes from the UART RX register to a RAM buffer automatically. Without the idle-line interrupt, the CPU would only be notified when the DMA buffer is **completely full** (e.g., 256 bytes).

But real-world messages are **variable length**. If a sensor sends 23 bytes, the DMA buffer is only partially filled. The CPU would wait forever for the remaining 233 bytes that never come.

The **idle-line interrupt** fires when the RX line has been HIGH (idle) for one full frame time after the last received byte — meaning "the message is done." The ISR then reads the DMA transfer counter to determine how many bytes were actually received, and processes the partial buffer.

This is the `UART_RX_RDY` event in Zephyr's async UART API. The same DMA-with-idle-detection pattern is used for SPI slave reception in your 100Hz bridge project.

</details>

---

**A7.** What is the difference between the TXE (TX Empty) and TC (Transmit Complete) interrupts on an STM32 UART? When does the distinction matter in practice?

<details><summary>Answer</summary>

- **TXE (TX register Empty):** The TX holding register is empty and can accept a new byte. But the **previous byte may still be shifting out** through the TX shift register onto the wire.
- **TC (Transmit Complete):** The shift register has finished transmitting the last bit of the last byte. The wire is now idle.

The distinction matters critically for **RS-485 half-duplex**: After the last byte, you must switch the DE (Driver Enable) pin from transmit to receive mode. If you switch on TXE, the last byte gets **truncated** because it's still shifting out. You must wait for **TC** before toggling DE.

It also matters for any protocol where you need to know that all data has physically left the chip (e.g., before entering sleep mode).

</details>

---

**A8.** Your Zephyr debug console suddenly shows garbage characters. List 3 things you would check, in order of likelihood.

<details><summary>Answer</summary>

1. **Baud rate mismatch:** The most common cause. Verify that the terminal emulator (minicom, picocom) is set to the same baud rate as the STM32 UART config (typically 115200). A wrong baud rate produces consistent garbage.

2. **TX/RX wires swapped:** TX of the STM32 must connect to RX of the USB-UART bridge, and vice versa. Swapped wires produce either garbage or complete silence.

3. **Data format mismatch:** Both sides must agree on data bits, parity, and stop bits. Most Zephyr setups use 8N1 (8 data, no parity, 1 stop). If the terminal is set to 7E1 or 8E1, you'll see intermittent wrong characters.

Less likely but worth checking: missing GND wire, wrong logic level (5V UART into 3.3V MCU), or broken/noisy USB-UART bridge.

</details>

---

**A9.** At 1 Mbaud (1,000,000 baud) with 8N1, a byte-per-byte interrupt-driven UART RX handler generates 100,000 interrupts per second. If each interrupt has ~1 µs overhead, what percentage of CPU time is consumed? How does DMA fix this?

<details><summary>Answer</summary>

- 100,000 interrupts × 1 µs = 100,000 µs = **100 ms per second = 10% CPU overhead** just for UART receive handling.
- On a Cortex-M7 running at 480 MHz, this is tolerable but wasteful — especially if you have other real-time tasks.

**DMA fix:** The DMA controller transfers bytes directly from the UART RX register to RAM without CPU involvement. The CPU is interrupted only when:
- The buffer is full (e.g., every 256 bytes → ~390 interrupts/sec), or
- The idle-line interrupt fires (end of message)

This drops CPU overhead from 10% to approximately **0.04%** — a 250× improvement. This is why DMA is essential for any high-speed serial communication, and the same principle applies to your SPI bridge.

</details>

---

## Section B — Spot the Bug

**B1.** A student connects an RS-485 sensor to an STM32 via a MAX485 transceiver. The DE and RE̅ pins are tied together and connected to a GPIO. Their code sets the GPIO HIGH before transmitting, then immediately sets it LOW in the TXE interrupt callback. Occasionally the last byte of each message is corrupted on the bus. What's wrong?

<details><summary>Answer</summary>

**Bug:** The student switches from TX to RX mode on the **TXE** (TX Empty) interrupt instead of the **TC** (Transmit Complete) interrupt.

TXE fires when the holding register is empty, but the shift register is still clocking out the last byte. Setting DE LOW at this point **disconnects the driver mid-byte**, truncating or corrupting the final byte on the bus.

**Fix:** Wait for the **TC** interrupt before switching DE LOW. TC fires only after the last bit (including the stop bit) has fully shifted out onto the wire.

</details>

---

**B2.** Two devices are communicating at "115200 baud." Device A uses a precision 12 MHz crystal with a divisor that achieves exactly 115200. Device B uses an internal RC oscillator rated at ±3% accuracy. The RC oscillator is currently running 4% fast. Communication works most of the time but produces ~1 corrupted byte per 50 bytes. Is this expected?

<details><summary>Answer</summary>

**Bug:** 4% baud rate error exceeds the ±2% tolerance for reliable 8N1.

With 4% error, the sampling drift per frame is:
- 10 bits × 0.04 = **0.4 bit-times** of drift per byte

At 0.4 bit-times, the sample point for the last data bit (D7) is 40% into the adjacent stop bit. This is very close to the edge — some bytes will be sampled correctly (when noise/jitter is favorable) and some won't.

The observed ~2% byte error rate (1 in 50) is consistent with marginal timing: most of the time the sample falls barely inside the correct bit, but noise occasionally pushes it over the boundary.

**Fix:** Use an external crystal or ceramic resonator for Device B (±0.01% accuracy), or switch to a lower baud rate where the percentage error is smaller relative to the bit time.

</details>

---

**B3.** A binary protocol sends sensor data over UART at 921600 baud with software flow control (XON/XOFF) enabled. The sensor periodically sends a packet containing an int16 reading. When the reading is exactly 4864 (0x1300), the receiver stops getting data for several seconds, then resumes. Why?

<details><summary>Answer</summary>

**Bug:** 0x1300 contains the byte **0x13**, which is the **XOFF** character. When the UART driver sees 0x13 in the data stream, it interprets it as a flow control command and tells the transmitter to **stop sending**.

The transmitter pauses until it either receives an XON (0x11) character, or times out and retries. The "several seconds" delay is the timeout/recovery period.

**Fix:** **Never use XON/XOFF flow control with binary data.** Either:
1. Disable flow control entirely (if the receiver can keep up)
2. Use **hardware flow control** (RTS/CTS) — these use dedicated wires, not in-band bytes
3. Use a framing protocol that escapes special bytes (e.g., SLIP or COBS encoding)

</details>

---

**B4.** An engineer calculates the throughput of their UART link: "My baud rate is 115200, so I can transfer 115200 bytes per second." They're surprised when their file transfer achieves only ~11,000 bytes/sec. Where's the error in their reasoning?

<details><summary>Answer</summary>

**Bug:** Confusing **baud rate** (bits per second) with **byte rate** (bytes per second).

At 115200 baud with 8N1 framing:
- Each byte requires **10 bit-times** on the wire (1 start + 8 data + 1 stop)
- Actual byte throughput = 115200 / 10 = **11,520 bytes/sec**

The 20% overhead from start and stop bits is inherent to UART framing. Additionally, if they're using a file transfer protocol with acknowledgements, headers, and retransmissions, effective throughput will be even lower.

**Rule:** UART throughput (bytes/sec) = baud rate / (1 + data_bits + parity_bits + stop_bits). For 8N1: baud / 10.

</details>

---

**B5.** A student wires up two STM32 boards for UART communication. They connect TX→TX and RX→RX between the boards. They see nothing on either side. What's wrong?

<details><summary>Answer</summary>

**Bug:** UART wiring must be **crossed**: TX of one device connects to **RX** of the other, and vice versa.

- TX→TX means both transmitters are driving the same wire (contention — both outputting, nobody listening)
- RX→RX means both receivers are listening to a wire that nobody is driving

**Fix:** Wire TX(A)→RX(B) and RX(A)→TX(B). Also ensure **GND is connected** between the two boards — without a common ground reference, the signal voltages are meaningless.

This is the #1 first-time UART wiring mistake. If you see "nothing at all," swap TX and RX before anything else.

</details>

---

**B6.** An RS-485 bus has 4 devices on it but only one 120Ω termination resistor (at one end). At 9600 baud on the bench, everything works perfectly. At 500 kbaud in the factory, random bytes are corrupted. What's going on?

<details><summary>Answer</summary>

**Bug:** Missing **second termination resistor** at the other end of the bus.

RS-485 requires **120Ω at each physical end** of the bus (matching the 120Ω characteristic impedance of the twisted pair cable). With only one termination:
- At 9600 baud (bit time = 104 µs), the signal transitions are slow enough that reflections from the unterminated end have time to settle before the sampling point → works fine on the bench.
- At 500 kbaud (bit time = 2 µs), reflections from the unterminated end bounce back and forth, creating **ringing** that crosses the differential threshold → phantom edges → bit errors.

**Fix:** Add a 120Ω resistor between A and B at the far end of the bus. Verify by measuring ~60Ω between A and B with power off (two 120Ω in parallel).

</details>

---

## Section C — Fill in the Blank / From Memory

### C1: UART Frame Format

Fill in the blanks for an 8N1 UART frame:

| Field | Logic Level | Duration | Purpose |
|-------|-------------|----------|---------|
| Idle | _____ | Indefinite | _____ |
| Start bit | _____ | 1 bit-time | _____ |
| Data bits | Variable | _____ bit-times | _____ |
| Parity (8N1) | N/A | _____ | _____ |
| Stop bit | _____ | 1 bit-time | _____ |

<details><summary>Answer</summary>

| Field | Logic Level | Duration | Purpose |
|-------|-------------|----------|---------|
| Idle | **HIGH (1)** | Indefinite | **Line rests at VCC when no data is being sent** |
| Start bit | **LOW (0)** | 1 bit-time | **Falling edge signals receiver to start sampling** |
| Data bits | Variable | **8** bit-times | **Payload, transmitted LSB first (D0 → D7)** |
| Parity (8N1) | N/A | **0 (no parity bit)** | **Not used in 8N1 — "N" means No parity** |
| Stop bit | **HIGH (1)** | 1 bit-time | **Returns line to idle, guarantees falling edge for next start bit** |

Total: 1 + 8 + 0 + 1 = **10 bit-times per byte**.

</details>

---

### C2: Baud Rates and Timing

Fill in the bit times and throughput:

| Baud Rate | Bit Time | Byte Rate (8N1) | Common Use |
|-----------|----------|-----------------|------------|
| 9600 | _____ µs | _____ B/s | _____ |
| 115200 | _____ µs | _____ B/s | _____ |
| 921600 | _____ µs | _____ B/s | _____ |
| 1,000,000 | _____ µs | _____ B/s | _____ |

<details><summary>Answer</summary>

| Baud Rate | Bit Time | Byte Rate (8N1) | Common Use |
|-----------|----------|-----------------|------------|
| 9600 | **104.2 µs** | **960 B/s** | **GPS NMEA, slow legacy sensors** |
| 115200 | **8.68 µs** | **11,520 B/s** | **Debug console, default for most MCU setups** |
| 921600 | **1.09 µs** | **92,160 B/s** | **High-speed embedded links** |
| 1,000,000 | **1.00 µs** | **100,000 B/s** | **Maximum standard UART** |

Formula: Bit time = 1/baud. Byte rate = baud / 10 (for 8N1).

</details>

---

### C3: Voltage Levels

Fill in the voltage ranges:

| Standard | Logic HIGH (Idle/1) | Logic LOW (Active/0) | Can Connect Directly to 3.3V MCU? |
|----------|--------------------|--------------------|-----------------------------------|
| TTL/CMOS (3.3V) | _____ | _____ | _____ |
| TTL/CMOS (5V) | _____ | _____ | _____ |
| RS-232 | _____ | _____ | _____ |
| RS-485 (differential) | _____ | _____ | _____ |

<details><summary>Answer</summary>

| Standard | Logic HIGH (Idle/1) | Logic LOW (Active/0) | Can Connect Directly to 3.3V MCU? |
|----------|--------------------|--------------------|-----------------------------------|
| TTL/CMOS (3.3V) | **3.3V** | **0V** | **Yes — native level** |
| TTL/CMOS (5V) | **5V** | **0V** | **Only if pin is 5V-tolerant; otherwise needs level shifter** |
| RS-232 | **-3V to -15V (typ. -12V)** | **+3V to +15V (typ. +12V)** | **NO — will destroy the pin. Use MAX3232.** |
| RS-485 | **A-B > +200mV** | **A-B < -200mV** | **No — requires RS-485 transceiver (MAX485)** |

Note: RS-232 is **inverted** compared to TTL — logic HIGH is a negative voltage!

</details>

---

## Section D — Lab / Calculation Tasks

**D1.** You're transmitting the letter `A` (ASCII 0x41 = binary 0100_0001) at 9600 baud, 8N1, TTL CMOS 3.3V. Draw the complete waveform on the TX line with voltage levels and durations. Label each bit.

<details><summary>Answer</summary>

0x41 = binary `0100_0001`. Transmitted **LSB first**: D0=1, D1=0, D2=0, D3=0, D4=0, D5=0, D6=1, D7=0.

Bit time = 1/9600 = 104.2 µs.

```
    3.3V ‾‾‾‾|    |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|    |‾‾‾‾‾‾‾‾‾‾‾‾
             |    |                                 |    |
    0V       |‾‾‾‾|    |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|    |
             
         IDLE START  D0   D1   D2   D3   D4   D5   D6   D7  STOP  IDLE
              (0)   (1)  (0)  (0)  (0)  (0)  (0)  (1)  (0)  (1)
             
    Time: 0   104  208  312  417  521  625  729  833  938  1042 µs
```

- **IDLE**: Line at 3.3V (logic HIGH)
- **START bit**: 0V for 104.2 µs (falling edge triggers receiver)
- **D0=1**: 3.3V for 104.2 µs
- **D1–D5=0**: 0V for 5 × 104.2 = 521 µs
- **D6=1**: 3.3V for 104.2 µs
- **D7=0**: 0V for 104.2 µs
- **STOP bit**: 3.3V for 104.2 µs (return to idle)

Total frame time: 10 × 104.2 = **1042 µs ≈ 1.04 ms**.

</details>

---

**D2.** Your STM32H7 runs its UART peripheral from a 100 MHz kernel clock. The peripheral divides this clock to generate the baud rate. What divisor is needed for 115200 baud? The divisor must be an integer. What is the actual baud rate achieved, and what is the percentage error?

<details><summary>Answer</summary>

With 16× oversampling (standard STM32 UART mode):

Divisor = f_clock / (16 × desired_baud) = 100,000,000 / (16 × 115,200) = 100,000,000 / 1,843,200 = **54.25**

Rounded to nearest integer: **54**

Actual baud rate = 100,000,000 / (16 × 54) = 100,000,000 / 864 = **115,740.74 baud**

Percentage error = |115,740.74 - 115,200| / 115,200 × 100% = 540.74 / 115,200 × 100% = **0.47%**

This is well within the ±2% tolerance. At 8× oversampling (available on STM32), the divisor granularity doubles, potentially allowing an even closer match.

</details>

---

**D3.** A ROS2 node on the Jetson publishes lidar data at 10 Hz, each message is 4000 bytes. You're forwarding this over UART to an external logging device. What minimum baud rate is needed? Include 8N1 overhead. Add 20% headroom for protocol framing.

<details><summary>Answer</summary>

Raw data rate: 4000 bytes × 10 Hz = **40,000 bytes/sec**

With 20% protocol headroom: 40,000 × 1.2 = **48,000 bytes/sec**

8N1 overhead: each byte needs 10 bit-times, so: 48,000 × 10 = **480,000 baud minimum**

The next standard baud rate above 480,000 is **500,000** (if supported) or **921,600** baud.

At 921,600 baud, available throughput = 92,160 bytes/sec, giving 92,160 / 48,000 = **1.92× headroom** — comfortable. You'd also want hardware flow control (RTS/CTS) or DMA to avoid data loss at this rate.

</details>

---

**D4.** You observe on an oscilloscope that the UART TX line transmits the following bit pattern (reading left to right in time): `LOW HIGH LOW HIGH LOW HIGH LOW HIGH LOW HIGH`. The baud rate is 115200. Decode the byte being transmitted (account for start/stop bits and LSB-first order).

<details><summary>Answer</summary>

Reading the bits left to right (in time order):

| Position | Level | Meaning |
|----------|-------|---------|
| 1 | LOW (0) | **Start bit** |
| 2 | HIGH (1) | D0 = 1 |
| 3 | LOW (0) | D1 = 0 |
| 4 | HIGH (1) | D2 = 1 |
| 5 | LOW (0) | D3 = 0 |
| 6 | HIGH (1) | D4 = 1 |
| 7 | LOW (0) | D5 = 0 |
| 8 | HIGH (1) | D6 = 1 |
| 9 | LOW (0) | D7 = 0 |
| 10 | HIGH (1) | **Stop bit** |

Data bits (LSB first): D0=1, D1=0, D2=1, D3=0, D4=1, D5=0, D6=1, D7=0

Reassembled (MSB first): D7..D0 = 0101_0101 = **0x55 = character 'U'**

This is the classic alternating pattern used for **auto-baud detection** — the perfect toggling signal makes it easy to measure the bit time.

</details>

---

**D5.** Your team is designing a binary protocol over UART between two STM32s at 1 Mbaud. Each message has a 2-byte header (0xAA55), 1-byte length, N payload bytes, and 2-byte CRC-16. If the payload is 48 bytes (IMU data), what is the maximum message rate you can sustain?

<details><summary>Answer</summary>

Total message size: 2 (header) + 1 (length) + 48 (payload) + 2 (CRC) = **53 bytes**

At 1 Mbaud, 8N1:
- Byte rate = 1,000,000 / 10 = **100,000 bytes/sec**
- Time per message = 53 / 100,000 = **0.53 ms**
- Max message rate = 100,000 / 53 = **1886.8 messages/sec**

This is well above the 100 Hz you need for your IMU bridge. Even at 115200 baud:
- Byte rate = 11,520 bytes/sec
- Max rate = 11,520 / 53 = **217.4 messages/sec** — still sufficient for 100 Hz, but with less headroom.

At 100 Hz with 53-byte messages, the bus utilization at 1 Mbaud would be only 53 × 100 / 100,000 = **5.3%** — plenty of margin.

</details>

---

## Section E — Deeper Thinking

**E1.** Your 100Hz STM32→Jetson bridge primarily uses SPI, but UART is available as a fallback. Compare UART and SPI for this specific use case: 52-byte IMU packets at 100Hz. Consider throughput, latency, CPU overhead, and failure modes. When would you fall back to UART?

<details><summary>Answer</summary>

**SPI advantages for this use case:**
- **Throughput:** At 10 MHz SPI, 52 bytes takes 52 × 8 / 10,000,000 = 41.6 µs. At 1 Mbaud UART, the same takes 520 µs — SPI is **12.5× faster**.
- **Latency:** SPI exchange is synchronous and predictable. UART adds start/stop bit overhead and potential FIFO delays.
- **CPU overhead:** Both can use DMA. But SPI slave DMA is event-driven (CS̄ assertion triggers transfer), while UART requires idle-line detection. SPI is slightly more deterministic.

**UART advantages:**
- **Simplicity:** No CS̄ timing issues, no "first byte garbage" slave problems, no CPOL/CPHA mismatch.
- **Distance:** UART (especially RS-485) works over longer wires than SPI.
- **Debugging:** You can tap a UART line with any USB-UART adapter and see the data in a terminal.

**When to fall back to UART:**
- SPI peripheral failure (hardware fault, corrupted DMA)
- Debug/diagnostics mode where you want human-readable output
- Long cable run between STM32 and Jetson that exceeds SPI signal integrity limits

At 100 Hz × 52 bytes = 5,200 bytes/sec, even 115200 baud UART (11,520 B/s) has 2× headroom — the fallback is viable, just with tighter margins.

</details>

---

**E2.** The 16× oversampling in UART receivers samples each bit 16 times and uses a majority vote from the middle 3 samples. Why 16× and not 8× or 32×? What's the engineering trade-off? How does this relate to the concept of "clock recovery" in SPI (which has an explicit clock)?

<details><summary>Answer</summary>

**Why 16×:**
- **Sampling resolution:** With 16 samples per bit, you can place the sampling window within ±1/16 = ±6.25% of the bit center. This provides good noise immunity while allowing ~3% baud rate mismatch.
- **8× oversampling** (available on STM32) doubles the maximum baud rate for a given peripheral clock, but halves the timing resolution — sampling is accurate to ±12.5% of bit center. Noise immunity is reduced, and baud rate tolerance tightens to ~2%.
- **32× oversampling** would improve resolution to ±3.1% of bit center but requires 2× the peripheral clock speed and consumes more power for diminishing returns.

**Majority vote from middle 3:** Instead of sampling at exactly the center (which might land on a noise spike), taking 3 adjacent samples near the center and using majority vote filters out brief noise glitches. This is cheaper than a full digital filter but substantially improves reliability.

**SPI comparison:** SPI doesn't need oversampling because it has an **explicit clock wire.** The receiver simply samples data on the specified clock edge — there's no ambiguity about when to sample. This is why SPI can run at much higher speeds (50+ MHz) than UART: it doesn't need to recover timing from the data signal itself. The trade-off is the extra wire (SCLK).

</details>

---

**E3.** Your warehouse robot uses RS-485 for motor controllers (4 motors on one bus) and UART for GPS. Both connect to the same STM32. If the motor bus runs at 500 kbaud with 10-byte commands at 1 kHz, and the GPS sends 82-byte NMEA sentences at 10 Hz — what is the total CPU interrupt load without DMA? With DMA? Make the case for why DMA is non-negotiable on a real-time robot controller.

<details><summary>Answer</summary>

**Without DMA (byte-by-byte interrupts):**

Motor bus (RS-485):
- 10 bytes × 1000 Hz = 10,000 bytes/sec sent
- Assume 10 bytes response × 1000 Hz = 10,000 bytes/sec received
- Total: 20,000 interrupts/sec

GPS (UART):
- 82 bytes × 10 Hz = 820 bytes/sec received
- 820 interrupts/sec

Total: **20,820 interrupts/sec**

At ~1 µs per interrupt (context save, ISR, restore): **20.8 ms/sec = 2.08% CPU** — sounds manageable, but:

- At 500 kbaud, bytes arrive every 20 µs. The ISR must complete in <20 µs or the next byte is lost (overrun).
- Any higher-priority interrupt (timer, SPI for your IMU bridge) that delays the UART ISR by >20 µs causes data loss.
- Motor commands are safety-critical — dropped bytes mean missed motor updates.

**With DMA:**
- Motor: 2 interrupts per cycle (TX complete + RX complete) × 1000 Hz = **2,000 interrupts/sec**
- GPS: 1 interrupt per sentence (idle-line) × 10 Hz = **10 interrupts/sec**
- Total: **2,010 interrupts/sec** — a **10× reduction**
- Each interrupt is also faster (just swaps buffer pointers, no byte handling)
- CPU overhead: < 0.2%

**Why DMA is non-negotiable:** On a real-time robot controller, the Cortex-M7 runs a control loop (PID, estimator, safety checks) at 1–10 kHz. Every microsecond of interrupt jitter degrades control quality. DMA moves serial I/O off the CPU entirely, ensuring the control loop gets predictable, low-jitter execution time. Without DMA, a burst of serial data can cause missed control deadlines — which on a robot means unstable motion or safety faults.

</details>

---

**E4.** UART, SPI, and I2C all solve the same fundamental problem: moving bytes between chips. Map out the design space — when does each protocol's trade-offs make it the clear winner? Consider your robot's full sensor suite: IMU (high speed, periodic), EEPROM (slow, infrequent), motor controllers (moderate speed, multi-drop), debug console.

<details><summary>Answer</summary>

| Protocol | Clock | Wires | Speed | Addressing | Best For |
|----------|-------|-------|-------|------------|----------|
| UART | None (async) | 2 + GND | Up to ~3 Mbaud | None (point-to-point) | Debug console, GPS, RS-485 multi-drop |
| SPI | Master provides | 4 + CS̄/slave | Up to ~100 MHz | CS̄ pin per slave | High-speed periodic: IMU, flash, displays |
| I2C | Master provides | 2 + pull-ups | 100k–3.4 MHz | 7-bit in-protocol address | Many slow devices: EEPROM, temp sensors, IO expanders |

**For your robot:**

- **IMU (ICM-42688) → SPI.** Needs 100 Hz × 52 bytes minimum, with low latency. SPI at 10–20 MHz gives microsecond-level reads. I2C at 400 kHz would work but is 25× slower and half-duplex.
- **EEPROM (calibration data) → I2C.** Reads/writes are infrequent (boot time, factory cal). I2C's 2-wire simplicity and built-in addressing fit perfectly. SPI would waste a CS̄ pin for something accessed once per boot.
- **Motor controllers (RS-485) → UART/RS-485.** Multi-drop bus up to 1200m, differential signaling for factory floor noise immunity. CAN is the superior choice here (Chapter 07), but RS-485 works for legacy systems.
- **Debug console → UART.** Human-readable protocol at 115200 via USB-UART bridge. Zero complexity, universal tool support. No other protocol makes sense for interactive debugging.

The key insight: there's no "best" protocol — each is optimal for a specific combination of speed, distance, device count, and complexity requirements.

</details>
