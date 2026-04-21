# 04 — UART Serial Deep Dive
### From electrons to bytes: the simplest serial protocol, inside and out
**Prerequisites:** Chapter 01 (basic voltage, pull-ups)
**Unlocks:** Debug consoles, GPS modules, RS-485 industrial sensors, DMA patterns

---

## Why Should I Care? (STM32/Jetson Project Context)

- The **debug console** on your STM32 Nucleo board uses UART over ST-Link (115200 baud)
- `printk()` and Zephyr shell output goes through UART
- Many industrial sensors (LIDAR, GPS, RS-485 actuators) speak UART
- Understanding UART DMA patterns transfers directly to SPI DMA patterns
- UART is the simplest serial protocol — if you understand it, SPI and I2C are easy extensions

---

# PART 1 — WHAT IS "SERIAL" COMMUNICATION?

---

## 1.1 Serial vs Parallel

**Parallel:** Send 8 bits simultaneously on 8 wires. Fast per-clock, but:
- 8 wires + clock + ground = 10+ wires
- At high speed, wires have slightly different propagation delays → bits arrive at different times (skew)
- Skew limits maximum clock rate to ~100MHz for short distances

**Serial:** Send 1 bit at a time on 1 wire. Fewer wires, no skew problem.
- Modern serial can run at GHz speeds (USB 3: 5Gbps, PCIe: 16Gbps)
- Won the "war" for almost everything except memory buses (DDR) and internal chip buses

```
    PARALLEL (old printer port, DDR memory):

    Data[0] ──────────────────── 
    Data[1] ──────────────────── 
    Data[2] ──────────────────── 
    Data[3] ──────────────────── 
    Data[4] ──────────────────── 
    Data[5] ──────────────────── 
    Data[6] ──────────────────── 
    Data[7] ──────────────────── 
    Clock   ──────────────────── 
    GND     ──────────────────── 

    10+ wires, all bits arrive "simultaneously" (but skew problems at speed)


    SERIAL (UART, SPI, USB):

    Data    ──────────────────── 
    (Clock  ──────────────────── )  ← optional, depends on protocol
    GND     ──────────────────── 

    2-4 wires total. Bits arrive one at a time. Simple, reliable.
```

---

## 1.2 Synchronous vs Asynchronous Serial

| Type | Clock Wire? | Examples | How Receiver Knows When to Sample |
|------|------------|----------|-----------------------------------|
| **Synchronous** | Yes (separate clock) | SPI, I2C | Samples on clock edges |
| **Asynchronous** | No clock wire | UART | Both sides agree on timing (baud rate) beforehand |

UART is **asynchronous** — there's no shared clock. Both sides must be configured to the same baud rate. If they disagree by more than ~2%, data gets corrupted.

---

# PART 2 — UART ELECTRICAL LEVELS

---

## 2.1 TTL / CMOS UART (What Your MCU Uses)

```
    Idle / Logic 1:   3.3V  (or 5V for 5V logic)
    Active / Logic 0: 0V

    This is "active low" for the start bit:
    the line sits HIGH when idle, drops LOW to signal "data coming"
```

## 2.2 RS-232 (What Old PCs Used)

```
    Idle / Logic 1:   -3V to -15V (typically -12V)
    Active / Logic 0: +3V to +15V (typically +12V)

    INVERTED compared to TTL! And much higher voltages!
```

**You CANNOT directly connect RS-232 to a 3.3V MCU. It will destroy the input pin.**

The MAX3232 level shifter converts between RS-232 (±12V) and CMOS (0/3.3V). It uses a charge pump with external capacitors to generate ±8V from 3.3V.

```
    RS-232 device ──── MAX3232 ──── STM32 UART pins
    (±12V)              (level      (0V / 3.3V)
                         shifter)
```

Nowadays, RS-232 is rare. Most "serial" connections are TTL UART through a USB-to-UART bridge (CP2102, CH340, FTDI FT232R). These ICs appear as a virtual COM port on your computer.

---

# PART 3 — UART FRAME FORMAT — BIT BY BIT

---

## 3.1 Anatomy of a UART Frame

```
    One UART frame (8N1 = 8 data bits, No parity, 1 stop bit):

    ┌─────┬────┬────┬────┬────┬────┬────┬────┬────┬──────┐
    │START│ D0 │ D1 │ D2 │ D3 │ D4 │ D5 │ D6 │ D7 │ STOP │
    │  0  │LSB │    │    │    │    │    │    │MSB │  1   │
    └─────┴────┴────┴────┴────┴────┴────┴────┴────┴──────┘
       ↑                                            ↑
    Falling edge                               Returns to
    triggers receiver                          idle (HIGH)
    to start sampling
```

**Step by step:**

1. **Idle:** Line sits at logic HIGH (3.3V)
2. **Start bit:** Transmitter pulls line LOW for exactly 1 bit-time. This falling edge tells the receiver "a byte is coming — start your clock NOW"
3. **Data bits:** 8 bits, transmitted **LSB first** (D0 before D7). Each bit is held for exactly 1 bit-time
4. **Parity bit (optional):** One extra bit for error detection. Even parity: total number of 1s (including parity) is even. Odd parity: total is odd. Most modern uses skip parity (8N1).
5. **Stop bit:** Line returns to HIGH for at least 1 bit-time. This guarantees a falling edge for the next start bit. Some protocols use 2 stop bits for extra margin.

**Total frame size:** 1 (start) + 8 (data) + 0 (no parity) + 1 (stop) = **10 bit-times per byte**

---

## 3.2 Worked Example: Transmitting 'U' (0x55) at 115200 Baud

The character 'U' is 0x55 = 0101_0101 in binary.

**Baud rate:** 115200 bits/second → 1 bit-time = 1/115200 = 8.68µs

```
    Binary of 0x55:  0 1 0 1 0 1 0 1
    Transmitted LSB first: D0=1, D1=0, D2=1, D3=0, D4=1, D5=0, D6=1, D7=0

    Voltage on TX line:
                                                            
    3.3V ┤‾‾‾‾‾│    │‾‾‾‾│    │‾‾‾‾│    │‾‾‾‾│    │‾‾‾‾‾‾‾
         │     │    │    │    │    │    │    │    │
    0V   ┤     │‾‾‾‾│    │‾‾‾‾│    │‾‾‾‾│    │‾‾‾‾│
         │                                              
         │START│ D0 │ D1 │ D2 │ D3 │ D4 │ D5 │ D6 │ D7 │STOP│
         │  0  │  1 │  0 │  1 │  0 │  1 │  0 │  1 │  0 │  1 │
         0   8.7  17  26   35  43   52   61  69   78   87  µs
```

Wait — let me redo the diagram. 0x55 = bits [0]=1,[1]=0,[2]=1,[3]=0,[4]=1,[5]=0,[6]=1,[7]=0:

```
    Voltage:
                     ___     ___     ___     ___
    3.3V ‾‾‾|   |‾‾|   |‾‾|   |‾‾|   |‾‾|   |‾‾‾‾‾‾‾
            |   |  |   |  |   |  |   |  |   |
    0V      |___|  |___|  |___|  |___|  |___|

           S  D0  D1  D2  D3  D4  D5  D6  D7  STOP
           0   1   0   1   0   1   0   1   0    1

    S = Start bit (always 0 = LOW)
    STOP = Stop bit (always 1 = HIGH)
    D0-D7 = data LSB first: 1,0,1,0,1,0,1,0 = 0x55

    Total time = 10 × 8.68µs = 86.8µs per byte
```

**Why 'U' (0x55) is special:** It creates a perfect alternating 0-1 pattern on the wire. This is why auto-baud detection circuits often ask you to send 'U' first — they measure the bit-time from the alternating edges.

---

## 3.3 Why LSB First?

UART transmits the **least significant bit first**. This seems backwards, but it has a practical reason: the receiver can start processing the data before all bits arrive. For example, "is the number odd?" can be answered after the first bit.

**But beware:** SPI is typically **MSB first** (configurable). When writing protocol bridges between UART and SPI devices, bit order differences can produce mystifying byte value swaps.

---

# PART 4 — BAUD RATE

---

## 4.1 What Baud Rate Is

Baud rate = number of bit-times per second. At 115200 baud, each bit occupies 1/115200 = 8.68µs.

The receiver uses the falling edge of the start bit to synchronize, then samples in the MIDDLE of each subsequent bit-time:

```
    TX line:   __|‾‾‾‾|____|‾‾‾‾|____|‾‾‾‾|...
                 ↑
                Start bit
                falling edge

    Receiver samples: (waits 1.5 bit-times from start edge)
                          ↓    ↓    ↓    ↓
    Sampling points:   __|‾‾‾‾|____|‾‾‾‾|____|...
                            ↑    ↑    ↑    ↑
                           D0   D1   D2   D3  ...
                         (sample in the middle of each bit)
```

**Key insight:** The receiver re-synchronizes on EVERY start bit. It does NOT maintain a free-running clock. This means baud rate errors accumulate within one frame (10 bits) but reset at the next byte.

---

## 4.2 Baud Rate Tolerance — What Happens When They Disagree

The receiver samples in the middle of each bit. If the clocks disagree, the sampling point drifts. After enough bits, it drifts past the edge of a bit-time → wrong bit sampled → corruption.

```
    BAUD RATE MISMATCH:

    TX at 115200:    |  D0  |  D1  |  D2  |  D3  |  D4  |  D5  |  D6  |  D7  |
    RX at 112000:    | D0   | D1   | D2   | D3   | D4   | D5   | D6   |  D7  X|
                                                                         ↑
                                                            Sample drifts into
                                                            adjacent bit → error!
```

**Calculating when corruption starts:**

```
    Mismatch per bit = |f_tx - f_rx| / f_tx
    
    At 3% mismatch: error per bit = 0.03 × 1 bit-time = 0.03 bit-times
    After N bits: total drift = N × 0.03
    Corruption at: drift > 0.5 (sampling past the bit boundary)
    N = 0.5 / 0.03 = 16.7 bits
    
    A 10-bit frame (8N1) → safe (drift = 10 × 0.03 = 0.3 bit-times)
    A 20-byte message = 200 bits → BUT! Re-sync at each byte start.
    So: per-byte drift = 10 × 0.03 = 0.3 bit-times → marginal but works.
    
    At 5% mismatch:
    Per-byte drift = 10 × 0.05 = 0.5 → right at the edge → unreliable!
```

**Rule of thumb:** Keep baud rate error below ±2% for reliable communication. Most MCU UARTs derive baud from their clock: if the clock crystal is ±0.01%, this is trivially met. Problems arise with:
- RC oscillators (±2-5% on some cheap MCUs)
- USB-UART bridges with non-standard baud rates

---

## 4.3 Common Baud Rates

| Baud Rate | Bit Time | Bytes/sec | Use Case |
|-----------|----------|-----------|----------|
| 9600 | 104µs | 960 | Slow sensors, GPS NMEA, legacy |
| 115200 | 8.68µs | 11,520 | Debug console, common default |
| 921600 | 1.09µs | 92,160 | High-speed embedded |
| 1,000,000 (1M) | 1.00µs | 100,000 | Maximal standard UART |
| 3,000,000 (3M) | 0.33µs | 300,000 | STM32 LPUART, some bridges |

**Throughput is not baud rate!** At 115200 baud with 8N1 framing:
```
    Effective data rate = 115200 / 10 bits per byte = 11,520 bytes/sec
    The 2 overhead bits (start + stop) cost you 20% of the bandwidth.
```

---

# PART 5 — UART HARDWARE

---

## 5.1 Inside the UART Peripheral

```
    ┌─────────────────────────────────────────────────┐
    │                    UART Peripheral               │
    │                                                   │
    │  TX Path:                                         │
    │  CPU writes byte → [TX Holding Reg] → [TX Shift Reg] → TX pin
    │                                          ↑ clocked by baud generator
    │                                                   │
    │  RX Path:                                         │
    │  RX pin → [RX Shift Reg] → [RX Holding Reg] → CPU reads byte
    │              ↑ clocked by 16× baud generator      │
    │              (samples 16 times per bit-time,       │
    │               takes the middle sample)             │
    │                                                   │
    │  FIFO (optional):                                 │
    │  [TX FIFO] → multiple bytes buffered for TX       │
    │  [RX FIFO] → multiple bytes buffered for RX       │
    │  STM32: FIFO depth varies by UART instance        │
    │                                                   │
    │  Interrupts: TXE (TX empty), RXNE (RX not empty), │
    │              IDLE (line idle after last byte),     │
    │              ORE (overrun error — RX not read      │
    │                   before next byte arrived)        │
    └─────────────────────────────────────────────────┘
```

**The 16× oversampling:** The UART peripheral doesn't just sample once per bit. It samples 16 times per bit-time and uses a majority vote from the middle 3 samples. This improves noise immunity.

---

## 5.2 Flow Control

### Hardware Flow Control (RTS/CTS)

Two extra wires:
- **RTS (Request to Send):** "I'm ready to receive data" — asserted by receiver
- **CTS (Clear to Send):** "You may send to me" — the other side's RTS, wired to my CTS input

```
    Device A                              Device B
    TX ──────────────────────────────── RX
    RX ──────────────────────────────── TX
    RTS ─────────────────────────────── CTS
    CTS ─────────────────────────────── RTS
    GND ─────────────────────────────── GND

    When Device B's RX buffer is almost full:
    B de-asserts RTS → A sees CTS de-asserted → A stops sending
    When B processes data and buffer has space:
    B re-asserts RTS → A sees CTS asserted → A resumes
```

**When it's needed:** High-speed UART (≥921600) with bursts of data that can exceed buffer size. At 115200 for debug logging, usually not needed.

### Software Flow Control (XON/XOFF)

The receiver sends special bytes in-band:
- **XOFF (0x13):** "Stop sending!" — receiver's buffer is full
- **XON (0x11):** "Resume sending!" — receiver has space

**Why it fails for binary data:** If your data happens to contain 0x13, the transmitter interprets it as flow control and stops sending. **Never use XON/XOFF for binary protocols.** Use hardware flow control or a framing protocol that escapes special bytes.

---

# PART 6 — DMA WITH UART

---

## 6.1 Why DMA Matters

**Without DMA (interrupt per byte):**

```
    At 1,000,000 baud (1Mbaud), 8N1:
    Bytes/sec = 100,000
    → 100,000 interrupts per second
    → Each interrupt takes ~1µs overhead (save registers, branch, restore)
    → 100,000 × 1µs = 100ms/sec = 10% CPU just handling UART interrupts!
```

**With DMA (interrupt per buffer):**

```
    DMA transfers bytes directly from UART RX register to RAM buffer.
    CPU is interrupted only when:
    1. Buffer is full (e.g., every 256 bytes → 390 interrupts/sec)
    2. IDLE line detected (no more bytes coming → grab partial buffer)

    CPU overhead: ~0.04% instead of 10%. The DMA controller does the work.
```

### 6.2 The Idle Line Interrupt Pattern (STM32)

For variable-length messages, you can't wait for a full buffer. The idle-line interrupt fires when the RX line has been HIGH (idle) for one full frame time after receiving data.

```
    RX data:     [byte][byte][byte][byte]...[last byte]___IDLE___
                                                        ↑
                                                    Idle-line interrupt fires
                                                    "Message is complete"

    DMA has already copied all bytes to RAM buffer.
    Interrupt handler reads how many bytes DMA transferred.
```

```c
/* Zephyr DMA UART pattern (conceptual): */
const struct device *uart = DEVICE_DT_GET(DT_NODELABEL(usart3));

uint8_t rx_buf[256];

/* Configure UART for async (DMA) mode */
uart_callback_set(uart, uart_callback, NULL);
uart_rx_enable(uart, rx_buf, sizeof(rx_buf), 100 /* timeout_us */);

/* Callback fires when: buffer full, timeout, or error */
void uart_callback(const struct device *dev, struct uart_event *evt, void *data) {
    if (evt->type == UART_RX_RDY) {
        /* evt->data.rx.buf + evt->data.rx.offset → received data */
        /* evt->data.rx.len → number of bytes */
        process_data(evt->data.rx.buf + evt->data.rx.offset, evt->data.rx.len);
    }
}
```

---

# PART 7 — RS-485: HALF-DUPLEX DIFFERENTIAL UART

---

## 7.1 Why RS-485?

UART (TTL) is point-to-point and limited to ~1 meter. RS-485 adds:
- **Differential signaling:** Two wires (A, B), measures voltage difference → noise rejection
- **Multi-drop:** Up to 32 devices on the same two wires
- **Long distance:** Up to 1200 meters at low baud rates
- **Half-duplex:** Only one device transmits at a time (they share the bus)

```
    RS-485 bus topology:

    ┌────┐  ┌────┐  ┌────┐  ┌────┐
    │Dev1│  │Dev2│  │Dev3│  │Dev4│
    └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘
      │        │        │        │
    ──┼────────┼────────┼────────┼──  Wire A
    ──┼────────┼────────┼────────┼──  Wire B
      │                              │
    ┌┴┐ 120Ω                    ┌┴┐ 120Ω  ← termination at BOTH ends
    └┬┘                         └┬┘
      A──B                        A──B
```

## 7.2 RS-485 Transceiver (MAX485 / SN65HVD75)

```
    MCU                 RS-485 Transceiver              Bus
    TX ──────── DI                         A ──────── 
    RX ──────── RO                         B ──────── 
    GPIO ────── DE/RE̅                      GND ────── 

    DE (Driver Enable): HIGH = transmitting, LOW = receiving
    RE̅ (Receiver Enable): LOW = receiving, HIGH = disabled
    
    Tie DE and RE̅ together → one GPIO controls direction:
    GPIO HIGH → transmit mode
    GPIO LOW  → receive mode
```

**Critical timing:** After transmitting the last byte, you must wait until the byte has fully shifted out the TX register before switching DE LOW (back to receive mode). If you switch too early, the last byte gets truncated. Use the UART TC (Transmit Complete) interrupt, not TXE (TX Empty — buffer empty, but byte still shifting out).

---

# PART 8 — USB-TO-UART BRIDGES

---

The CP2102, CH340, and FTDI FT232R are ICs that present as a USB device to the host computer and translate USB data to/from UART signals. They contain:
- USB transceiver
- Internal clock synthesis (no external crystal on CP2102)
- UART engine with configurable baud rate
- Optional RTS/CTS/DTR lines

From the MCU's perspective, it's just talking UART. From the PC side, it's a virtual COM port.

```
    PC ──[USB cable]── CP2102 ──[TX/RX wires]── STM32 UART
    
    On PC:  /dev/ttyUSB0 (Linux) or COM3 (Windows)
    Terminal: minicom, picocom, PuTTY, or VS Code Serial Monitor
```

---

# PART 9 — ASCII vs BINARY PROTOCOLS

---

## 9.1 ASCII/Text Protocols

Human-readable. Each value is transmitted as ASCII characters:

```
    Sending the number 1234:
    Bytes on wire: '1' '2' '3' '4' '\r' '\n'
    Hex:           0x31 0x32 0x33 0x34 0x0D 0x0A
    
    6 bytes for a 16-bit number! Inefficient but easy to debug with a terminal.
```

Examples: GPS NMEA (`$GPGGA,123519,4807.038,N,...`), AT commands (`AT+OK\r\n`), Zephyr shell.

## 9.2 Binary Protocols

Values transmitted as raw bytes:

```
    Sending the number 1234 (0x04D2) in binary:
    Bytes on wire: 0x04 0xD2
    
    2 bytes! But if you look at it in a terminal, you see garbage characters.
```

**The mixing bug:** If you send binary data through a system that interprets special ASCII characters:
- 0x0A in your binary data = '\n' → line-ending handling may eat it or add 0x0D
- 0x13 in your binary data = XOFF → software flow control stops transmission
- 0x00 in your binary data = null → C string functions terminate early

**Fix:** Use a proper framing protocol (header + length + payload + CRC) and disable any text-mode processing on the UART.

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| Garbage characters on terminal | Wrong baud rate | Try all common baud rates. Logic analyzer shows correct timing | Match baud rate (check both sides!) |
| Nothing received | TX and RX swapped | TX of device A must go to RX of device B, and vice versa | Swap TX/RX wires |
| Characters work sometimes | Baud rate mismatch >2% | Count error rate over 1000 bytes. Use scope to measure actual bit time | Fix clock source or baud divisor |
| Corruption in long messages | No flow control, RX buffer overflow | ORE (overrun) flag in UART status reg | Enable RTS/CTS or increase buffer/use DMA |
| Works on bench but not in field | Missing GND wire | Measure voltage between device GNDs — should be <0.5V | Connect GND between devices |
| ±12V on MCU pin → dead pin | RS-232 connected to TTL UART | Measure voltage at MCU pin with scope | Add MAX3232 level shifter |
| RS-485 bus unreliable | Missing termination resistors | Scope A-B: look for ringing after edges | Add 120Ω at each end of bus |
| Last RS-485 byte corrupted | DE switched before TX complete | Check timing of DE vs actual TX finish | Wait for TC interrupt, not TXE |
| Binary data truncated | XON/XOFF enabled, 0x13 in data | Disable XON/XOFF in UART config | Use hardware flow control or no flow control |
| Characters received but wrong | Wrong data bits, parity, stop bits | 8N1 vs 7E1, etc. Check both sides match | Match configurations exactly |
| Single byte OK, multi-byte fail | Missing level shifter → voltage marginal | Scope TX line: 3.3V sagging to 2.5V | Check wiring, power supply, add level shifter if needed |

---

# QUICK REFERENCE CARD

```
┌────────────────────────────── UART SERIAL CHEAT SHEET ──────────────────────────────────────┐
│                                                                                              │
│  FRAME FORMAT (8N1):  [START=0] [D0] [D1] [D2] [D3] [D4] [D5] [D6] [D7] [STOP=1]         │
│                       10 bit-times per byte, LSB first                                      │
│                                                                                              │
│  BAUD →  BIT TIME          THROUGHPUT (8N1)                                                 │
│  9600    104.2 µs          960 bytes/sec                                                    │
│  115200    8.68 µs         11,520 bytes/sec                                                 │
│  921600    1.09 µs         92,160 bytes/sec                                                 │
│  1000000   1.00 µs        100,000 bytes/sec                                                 │
│                                                                                              │
│  BAUD TOLERANCE: ±2% max for reliable 8N1                                                   │
│  DRIFT = bits_per_frame × (baud_error%) → must stay < 50% of 1 bit-time                    │
│                                                                                              │
│  WIRING:   TX(A) → RX(B),  RX(A) → TX(B),  GND(A) → GND(B)                               │
│                                                                                              │
│  VOLTAGE LEVELS:                                                                             │
│    TTL/CMOS:  HIGH=VCC (3.3V or 5V), LOW=0V                                                │
│    RS-232:    HIGH=-12V, LOW=+12V  (inverted, higher voltage)                               │
│    RS-485:    Differential: A-B > +200mV = 1, A-B < -200mV = 0                             │
│                                                                                              │
│  DMA PATTERN: uart_rx_enable() with double-buffer + idle-line timeout                       │
│  USB-UART:    CP2102, CH340, FTDI → virtual COM port on PC                                  │
│                                                                                              │
│  RS-485:  2-wire (A,B) + GND, 120Ω at each end, DE pin toggles TX/RX mode                 │
│  GOTCHA:  swap TX/RX first if nothing works. Then check baud. Then check GND.               │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
