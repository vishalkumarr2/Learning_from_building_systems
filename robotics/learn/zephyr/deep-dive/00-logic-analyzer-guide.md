# Logic Analyzer Discipline Guide
### Saleae Logic 2 — trigger setup, decoder configuration, and systematic waveform analysis

---

## Overview

A logic analyzer is the single most valuable tool for embedded debugging after a multimeter.
This guide covers how to use Saleae Logic 2 productively — not just how to click buttons,
but how to build discipline so you capture the right data the first time.

**This guide is referenced throughout sessions 01–11 wherever physical bus verification is needed.**

---

## 0. Step Zero — Verify the Bus Before Writing Firmware

**The most expensive mistake in embedded development:** spending two days debugging firmware
before discovering the hardware was never working.

Before writing a single line of peripheral driver code:

| Bus | Verification command | Expected output |
|-----|----------------------|-----------------|
| CAN | `sudo ip link set can0 up type can bitrate 500000 && candump can0` | Frames visible from other nodes |
| UART/GPS | `minicom -D /dev/ttyUSB0 -b 9600` | Raw NMEA sentences scrolling |
| I2C | `i2cdetect -y 1` | Device address visible in grid |
| SPI | Connect loopback (MOSI→MISO), run `spidev_test` | Echo bytes match |

**If the bus shows nothing at this stage, the problem is hardware — wiring, power, termination,
or chip selection. No firmware fix will help. Resolve it at this layer first.**

This is not optional. It is Step 0 for every session in this curriculum.

---

## 1. Hardware Setup

### 1.1 Channel Assignment (stick to this mapping)

Use a consistent channel numbering scheme across all projects. Consistency means your saved
`.logicdata` files are reusable and your mental model is stable.

```
CH0  — SCLK  (SPI clock / I2C SCL)
CH1  — CS    (SPI chip select, active low)
CH2  — MOSI  (SPI master-out / I2C SDA)
CH3  — MISO  (SPI master-in)
CH4  — TX    (Target TX → Saleae: what the TARGET SENDS, what the host receives)
CH5  — RX    (Target RX → Saleae: what the HOST SENDS, what the target receives)
CH6  — CAN_TX (before transceiver)
CH7  — trigger / user-defined
```

> ⚠️ **TX/RX direction reminder:** Names are from the **target’s** perspective.
> - Physically wire: `target TX pin` → `Saleae CH4` (you see what the target transmits)
> - Physically wire: `target RX pin` → `Saleae CH5` (you see what the host transmits to the target)
> The most common first-time mistake is swapping these. If the UART decoder shows no data or garbage, flip CH4/CH5.

**GND wire placement**: Run a ground wire from the Saleae GND pin to the target board GND
*at the point closest to your signal source*. A long GND loop introduces inductive coupling
that appears as glitches on the logic analyzer.

### 1.2 Probe Loading

Each Saleae probe tip presents ~10pF and a 1MΩ input impedance. On fast SPI signals
(≥10MHz), this can round edges enough to cause false captures. Use the Saleae hook tips
and keep leads short (<15cm).

For 20MHz+ SPI: solder a 220Ω series resistor at the target pad, probe after the resistor.

---

## 2. Sample Rate Selection

| Signal speed | Recommended sample rate | Why |
|---|---|---|
| UART 9600 baud | 1 MSPS | 100× oversampling |
| UART 115200 baud | 8–12 MSPS | Detect framing errors |
| I2C 100kHz | 8 MSPS | Capture ACK bits cleanly |
| I2C 400kHz fast-mode | 8–16 MSPS | |
| SPI 1MHz | 8 MSPS | |
| SPI 10MHz | 100 MSPS | Nyquist + margin |
| SPI 20MHz | 200 MSPS (Logic Pro 8 required) | |
| CAN 500kbps | 8 MSPS | |
| CAN 1Mbps | 16 MSPS | |

**Rule**: Capture at least 8× the bit rate. Lower than 4× causes frequent decode errors.

---

## 3. Trigger Configuration

### 3.1 The Fundamental Rule

**Always trigger on a CS falling edge for SPI.** Never trigger on SCLK — SCLK toggles
during every transfer and gives you random frame alignment.

For I2C: trigger on SDA falling edge while SCL is high (START condition).  
For UART: trigger on the falling edge of the start bit (TX or RX channel falling edge,
since idle state is HIGH).  
For CAN: trigger on a long dominant bit sequence (SOF + arbitration field).

### 3.2 Setting a Trigger in Logic 2

1. Add a channel in the sidebar
2. Click the trigger icon (lightning bolt) next to the channel
3. Select edge type:
   - **↓ Falling** — CS active-assertion, UART start bit
   - **↑ Rising** — CS deassert (end of transfer), useful for measuring transfer time
   - **High** / **Low** — level triggers for stuck signals

### 3.3 Pre-trigger Buffer

Set a pre-trigger buffer of ~10ms when debugging intermittent faults. This lets you see
what was happening **before** the trigger condition — crucial for:
- CAN bus-off events (you want the frames that caused it, not the aftermath)
- SPI CS glitches (you want the clean transfers before the fault)
- ESD-induced resets

In Logic 2: Capture Settings → "Pre-trigger buffer" → 10–50ms.

### 3.4 Conditional Triggers (Logic 2 Protocol Triggers)

After adding a SPI decoder, you can trigger on a specific byte value:

1. Add SPI analyzer
2. In Trigger panel: Add → Protocol → SPI → MOSI value = `0xDE`

This is invaluable for finding a specific command byte in a high-frequency stream.

---

## 4. Decoder Configuration

### 4.1 SPI Decoder Settings — Common Mistakes

| Setting | Common mistake | Correct approach |
|---|---|---|
| CPOL | Default 0 | Read your MCU/sensor datasheet — CPOL=1 means idle-high clock |
| CPHA | Default 0 | CPHA=0 = sample on first edge, CPHA=1 = sample on second edge |
| CS polarity | Active-low assumed | Some SPI sensors use active-high CS |
| Bit order | MSB first assumed | Some sensors send LSB first |
| Enable line | Channel 1 (CS) | Must match your wiring |

**Verification**: After setting up the SPI decoder, look at the decoded bytes in the
"Data" column. If you see random-looking bytes but the waveform shows clean edges, you
have CPOL/CPHA wrong. Try all 4 mode combinations (Mode 0–3) if unsure.

### 4.2 UART Decoder Settings

```
Baud rate: must match exactly (9600, 115200, etc.)
Data bits: 8 (almost always)
Stop bits: 1
Parity: None (unless your target uses it)
Bit order: LSB first (UART standard)
```

**Inverted UART**: RS-232 signals are inverted (1 = negative voltage). If you probe an
RS-232 line with a 3.3V logic analyzer probe, enable "Inverted" in the decoder settings.
If your decoded bytes look like garbage but the waveform is clean, the idle state (HIGH
vs LOW) is wrong — try toggling inversion.

### 4.3 I2C Decoder Settings

I2C only needs two channels: SCL and SDA. The decoder automatically identifies:
- START/STOP conditions
- Address byte (7-bit + R/W bit)
- ACK/NACK bits
- Data bytes

**ACK vs NACK**: A NACK (logic 1 during ACK slot) appears as `[NAK]` in the decoded
stream. This is the primary signal for:
- Wrong I2C address
- Device not ready
- Peripheral power not applied

### 4.4 CAN Decoder Settings

```
Bit rate: 500000 (500kbps typical for AMR) or 1000000
Sample point: 75% (default works for most cases)
```

CAN uses differential signaling; most logic analyzers probe CAN_H relative to ground.
The CAN transceiver converts differential to single-ended — probe **after** the
transceiver at CANH or at the RX pin of your MCU.

**Tip**: If you see "bit error" in the decoded stream, your sample point or bit rate
is slightly off. Adjust the sample point by ±5% and re-decode.

---

## 5. Capturing and Analyzing an SPI Transaction

### 5.1 Workflow: STM32 SPI Slave Debug

This is the workflow for session 02 and beyond.

**Step 1: Verify idle state**
Before any data flows, with CS high:
- SCLK should be **low** (SPI mode 0) or **high** (SPI mode 3)
- MOSI/MISO can be any level
- If CS is floating (not high), your pull-up resistor is missing

**Step 2: Capture a single transaction**
- Trigger: CH1 (CS) falling edge
- Duration: 50ms
- Examine: count the SCLK pulses. For an 8-bit byte: 8 pulses. For 64 bytes: 512 pulses.

**Step 3: Verify byte count**
In Logic 2, after decoding: select all SPI annotations. The annotation count ÷ 2 = number
of bytes transferred (each byte generates one MOSI + one MISO annotation).

**Step 4: Verify CS timing**
Measure the time from CS falling edge to first SCLK edge. On STM32 SPI, this "CS setup
time" should be < 1µs. If it's > 5µs, your master has a slow GPIO initialization path.

**Step 5: Look for CS glitches**
Zoom in on the CS line between transactions. Any high-to-low-to-high pulse shorter than
one byte period is a glitch. These cause phantom transactions in your slave handler.

---

## 6. Timing Measurements

### 6.1 Measuring Transaction Duration

Use Logic 2 timing markers:
1. Press `A` key to drop Marker A at cursor position
2. Navigate to end of transaction
3. Press `B` key to drop Marker B
4. Logic 2 shows Δt between A and B in the toolbar

For a 64-byte SPI transfer at 10MHz:
- Expected: 64 × 8 bits / 10,000,000 Hz = 51.2µs
- Measure your actual value and compare

### 6.2 Inter-Frame Gap

CS deassertion to next CS assertion = inter-frame gap. For a 100Hz loop with 51.2µs
transfer, the gap should be 10ms - 51.2µs ≈ 9948µs. If the gap is 0 (back-to-back
transfers), your master doesn't respect the minimum CS deassert time.

### 6.3 SPI → ZBus → ROS2 Latency

To measure end-to-end latency:
1. Add a GPIO "timestamp" output on the STM32: toggle a pin at the start of DMA TX
2. Probe that pin on CH7 of the logic analyzer
3. Measure CH7 pulse → CS assertion on CH1 = STM32-side processing latency
4. The ROS2 end can't be measured on a logic analyzer — use `ros2 topic echo --once`
   with `header.stamp` comparison

---

## 7. UART GPS Capture Workflow

For session 08 (UART GPS NMEA):

**Step 1: Verify baud rate**
Before decoding, measure the width of the narrowest pulse in the UART data stream.
For 9600 baud: 1 bit = 1/9600 = 104.2µs. If your measurement shows 104µs, the baud rate
is correct.

**Step 2: Find NMEA sentence boundaries**
In the decoded stream, look for the `$` character (0x24) — it marks the start of every
NMEA sentence. The CRLF (0x0D, 0x0A) marks the end.

**Step 3: Measure sentence duration**
Trigger on the `$` start. Measure to the final `\n`. For `$GNGGA` at 9600 baud:
- Sentence is ~70 characters × (1 start + 8 data + 1 stop) bits = 700 bits
- At 9600 baud: 700 / 9600 ≈ 72.9ms

This tells you that at 9600 baud, the GPS sentence takes 73ms to arrive — relevant
for understanding why ring-buffer accumulation is necessary.

---

## 8. CAN Bus Capture Workflow

For session 07 (CAN bus encoder):

**Step 1: Verify bus state**
Idle CAN bus: CANH ≈ 2.5V, CANL ≈ 2.5V (recessive). Probe CAN_H relative to GND.
At logic level: recessive = 1, dominant = 0.

**Step 2: Capture CAN frames**
Trigger: falling edge on CH6 (CAN_TX from MCU, before transceiver).
Look for the Start Of Frame bit (always dominant = 0) followed by 11-bit arbitration ID.

**Step 3: Verify frame contents**
After enabling CAN decoder: the annotation should show:
```
ID: 0x201  DLC: 8  Data: [FF D8 00 C8 AB CD EF 01]  ACK
```
If you see `Error: stuff bit` annotations, your bitrate setting in the decoder is wrong
or there's a physical problem (termination).

**Step 4: Look for error frames**
CAN error frames appear as a 6-bit dominant sequence after a bit error. Logic 2 marks
these as `[Error frame]`. If you see more than 1 error per 100 frames, investigate the
physical layer (termination, cabling, ground loops).

---

## 9. Saving and Annotating Captures

### 9.1 Always Save Captures

Save every non-trivial capture as `.logicdata`:
- `debug_<date>_<project>_<what>.logicdata`
- Example: `debug_20260412_spi-slave_cs-glitch.logicdata`

Logic 2 files include all channel data, decoders, and timing markers.

### 9.2 Export for Documentation

To embed a capture in a bug report:
1. Logic 2 → File → Export Data → CSV (for analysis in Python/Excel)
2. Logic 2 → Screenshot → Copy to clipboard (for quick paste into Slack/ADO)

For session documentation: export a CSV of decoded SPI bytes and include the hex dump
in your notes to prove the framing is correct.

---

## 10. Practical Gotchas Reference

| Symptom | Likely cause | Check |
|---|---|---|
| All decoded bytes are `0xFF` | MISO/MOSI swapped | Swap CH2 and CH3 leads |
| Decoder shows `Error: Frame error` | Wrong CPOL/CPHA | Try SPI Mode 1, 2, 3 |
| UART shows garbage but waveform clean | Wrong baud rate | Measure bit width manually |
| CAN shows no frames, candump silent | Missing termination OR wrong bitrate | Measure CANH-CANL with multimeter |
| Logic shows CS glitch < 1µs | PCB ground bounce | Add 100nF cap near CS driver pin |
| SPI bytes look right but STM slave misses first 1-2 bytes | DMA not armed before CS | See session 02 arming sequence |
| I2C shows [NAK] on every address | Device not powered or wrong address | Check VCC on target, read datasheet for address pins |
| Trigger fires but capture is empty | Pre-trigger buffer too small | Increase to 50ms and retrigger |
