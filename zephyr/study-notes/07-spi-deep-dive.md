# SPI Deep Dive — Study Notes
### Wire-Level Protocol, All 4 Clock Modes, Multi-Slave Topologies, DMA Mechanics, and Failure Modes
**Context:** Python/ROS developer learning embedded systems · assumes zero prior hardware protocol experience

---

## PART 1 — ELI15 Concept Explanations

---

### §1 — SPI Electrical Fundamentals

---

#### 1.1 Push-Pull Outputs — Why SPI Is NOT Like I2C

**The analogy: shouting vs whispering**

I2C uses **open-drain** outputs. Think of an open-drain pin as someone who can only
pull a rope DOWN (to ground). To get the line HIGH, you tie a resistor from the line
to VCC (a pull-up). When nobody pulls down, the pull-up gently brings it high. This
is great for multi-master buses because two devices pulling down simultaneously won't
damage anything — they're both pulling the same direction.

SPI uses **push-pull** outputs. A push-pull pin actively DRIVES the line both HIGH
(connects to VCC through a transistor) and LOW (connects to GND through a transistor).
It doesn't whisper — it shouts. This is why SPI can run at 50 MHz while I2C tops out
at 3.4 MHz in its fastest mode. The strong drive means sharp edges, fast transitions,
and no slow pull-up to wait for.

**But there's a dangerous consequence:**

If TWO push-pull outputs try to drive the same wire to different levels — one to VCC,
one to GND — you get a **short circuit through the transistors**. Current flows from
VCC through device A's high-side transistor, through the wire, through device B's
low-side transistor, straight to GND. This is called **bus contention** and it:
- Produces excessive current (10-50 mA continuous)
- Creates a voltage somewhere between VCC and GND (neither valid HIGH nor LOW)
- Causes data corruption at minimum, chip damage at maximum

This is exactly why **only one slave may drive MISO at any time**. The chip-select
signal (CS) is what enforces this — a slave must tristate (disconnect, go high-impedance)
its MISO pin the instant CS goes HIGH.

```
Push-pull output (MOSI, MISO, CLK):

  VCC ──┐
        │
      ┌─┴─┐
      │P-FET│ ← ON when output=HIGH
      └─┬─┘
        ├──── output pin
      ┌─┴─┐
      │N-FET│ ← ON when output=LOW
      └─┬─┘
        │
  GND ──┘

  Contrast with open-drain (I2C):

  VCC ──┐
        │
       [R]  ← external pull-up resistor
        │
        ├──── output pin
      ┌─┴─┐
      │N-FET│ ← ON when output=LOW
      └─┬─┘       (no P-FET — can ONLY drive low)
        │
  GND ──┘
```

---

#### 1.2 CS (Chip Select) — The Hardware Addressing Mechanism

**The analogy: a teacher calling on students by name**

In a classroom, only the student whose name is called should answer. In SPI, every
slave has a CS (Chip Select) pin. The master pulls ONE slave's CS LOW to "call on" that
slave. All other slaves see their CS HIGH and stay SILENT — specifically, they tristate
their MISO output (go high-impedance, as if their pin is disconnected).

Key facts about CS:

| Aspect | Detail |
|--------|--------|
| Active level | LOW by convention (assert = pull to GND) |
| Who drives it | Always the MASTER |
| When to assert | Before the first CLK edge of a transaction |
| When to deassert | After the last CLK edge (some devices: between bytes) |
| What happens if forgotten | MISO floats → reads as 0xFF (with pull-up) or 0x00 (without) |

**Why CS is "active low":**

Historical convention from TTL logic. A pull-up resistor on CS means that when no
master is connected, the slave sees CS=HIGH and stays disabled. This is a **safe
default** — the slave won't randomly drive MISO and cause contention. Many modern
devices allow configuring active-high CS, but active-low is the 95% case.

**CS timing parameters (from a typical datasheet):**

```
  CS:   ‾‾‾‾‾‾\________________________________/‾‾‾‾‾‾
                ← t_CSS →                ← t_CSH →
  CLK:  ‾‾‾‾‾‾‾‾‾‾‾‾_/‾\_/‾\_..._/‾\_‾‾‾‾‾‾‾‾‾‾‾‾‾
                     ↑ first CLK edge    ↑ last CLK edge

  t_CSS = CS Setup time: minimum time CS must be LOW before first CLK edge
  t_CSH = CS Hold time:  minimum time CS must remain LOW after last CLK edge
```

Typical values: t_CSS = 5-50ns, t_CSH = 10-100ns. Violating these causes the
first or last byte to be garbage — the slave wasn't ready or released too early.

---

#### 1.3 Voltage Levels — 3.3V vs 5V and the Contention Problem

Most modern microcontrollers run at 3.3V. But many legacy peripherals, Arduinos, and
industrial sensors still use 5V SPI. You CANNOT directly connect them without damage
or data corruption.

**The risks:**

| Scenario | Problem |
|----------|---------|
| 5V master → 3.3V slave | MOSI/CLK at 5V exceeds slave's absolute max (usually 3.6V). Damages slave input pins. |
| 3.3V master → 5V slave | MOSI/CLK at 3.3V may not reach slave's V_IH threshold (often 3.5V for 5V logic). Slave misreads bits. |
| Mixed voltage MISO | 5V slave drives MISO to 5V. 3.3V master's GPIO sees overvoltage. Damage or latch-up. |

**Solutions:**

1. **Bidirectional level shifter** (e.g., TXB0104): actively translates voltage levels.
   Adds propagation delay (1-4ns per stage). Good for <20 MHz.

2. **Series resistor + clamp diode**: cheap, works for unidirectional signals (MOSI, CLK).
   Not suitable for bidirectional lines without more circuitry.

3. **5V-tolerant inputs**: some 3.3V MCUs have 5V-tolerant GPIO pins (marked "FT" in
   STM32 datasheets). This handles incoming 5V on MISO without a shifter. Still need
   a shifter for outgoing MOSI/CLK if the slave needs 5V thresholds.

**The MISO contention problem with level shifters:**

Level shifters have a drive strength. If a slave is actively driving MISO LOW but the
level shifter is trying to pull it to 3.3V (because of its bus-hold circuitry), you
get a voltage fight. Always check that your level shifter supports tristate pass-through
for the MISO direction.

---

#### 1.4 Signal Integrity at Speed

**The analogy: shouting across a field vs across a canyon**

At 1 MHz SPI, one bit lasts 1000ns. A 10cm wire on a breadboard adds ~0.5ns of
propagation delay and some reflections — totally negligible. But at 10 MHz, one bit
lasts only 100ns, and suddenly that 10cm wire matters.

**The physics:**

A voltage edge on a PCB trace travels at roughly 15 cm/ns (about half the speed of
light, determined by the dielectric constant of FR4 copper-clad board material).
A 10 MHz clock has a 50ns half-period. In 50ns, the signal travels:

```
  distance = 15 cm/ns × 50 ns = 750 cm = 7.5 meters
```

That sounds fine — but reflections happen at impedance discontinuities (every
connector, via, right-angle trace, or unterminated wire end). The reflection travels
back and forth, creating **ringing** — the signal oscillates around the intended
level for a few nanoseconds after each edge transition.

```
  Ideal CLK edge:        Real CLK edge on long wire:

  VCC ──┐                VCC ──┐  ╭─╮
        │                      │ /   \
        │                      │/     \─── ringing
  GND ──┘                GND ──┘
        ↑                      ↑
     one clean             overshoot and
     transition            undershoot
```

If ringing crosses the input threshold TWICE, the receiving chip sees TWO clock edges
instead of one. This causes **double-clocking** — an extra bit gets shifted, and the
entire frame is shifted by one position. Every subsequent byte is corrupted.

**Practical guidelines by speed:**

| SPI Speed | Wire Type | Max Length | Notes |
|-----------|-----------|------------|-------|
| ≤1 MHz | DuPont jumpers OK | 30 cm | Breadboard-safe |
| 1-5 MHz | DuPont jumpers marginal | 15 cm | Works, but may glitch |
| 5-10 MHz | PCB traces required | 10 cm | Add ground plane, keep traces short |
| 10-25 MHz | Controlled impedance PCB | 5 cm | Series termination resistors (33Ω) |
| 25-50 MHz | Expert-level PCB design | 2 cm | Matched traces, ground stitching vias |

**Series termination resistors:**

Place a 22-47Ω resistor in series with CLK, MOSI, MISO — as close to the DRIVER
pin as possible (not at the receiver end). The resistor absorbs the reflected wave,
preventing ringing. The exact value depends on trace impedance (typically 50Ω for
microstrip), so 33Ω is a common starting point.

```
  Master CLK pin ──[33Ω]──── PCB trace ──── Slave CLK pin
                  ↑ series termination
                    placed at source
```

**Ground return path:**

Every signal needs a return current path. If your SPI clock runs on one side of a PCB
but the ground plane has a gap underneath, the return current must detour around the
gap. This adds inductance and increases ringing. **Rule: never route SPI signals
across ground plane splits.**

---

#### 1.5 Multi-Slave Topologies

SPI does not have software addressing like I2C (where each device has a 7-bit address).
Instead, CS IS the address — each slave gets its own CS pin, selected by the master
in hardware.

**Topology 1: Independent CS (most common)**

```
                    ┌──────────┐
  ┌──── MOSI ──────┤ Slave A  │
  │  ┌── MISO ─────┤ (CS=PA4) │
  │  │  ┌ CLK ─────┤          │
  │  │  │  ┌ CS_A ─┤  CS      │
  │  │  │  │       └──────────┘
  │  │  │  │
┌─┴──┴──┴──┴─┐     ┌──────────┐
│   Master   │     │ Slave B  │
│            ├─────┤ (CS=PA5) │
│ PA4 = CS_A │     │          │
│ PA5 = CS_B │     │  CS      │
│ PA6 = CS_C │     └──────────┘
└────────────┘
                    ┌──────────┐
          ...──────┤ Slave C  │
                   │ (CS=PA6) │
                   │  CS      │
                   └──────────┘
```

- **Wires needed**: N slaves → 3 shared (CLK, MOSI, MISO) + N CS = N+3 total GPIO pins
- **Advantage**: simple, each slave independently addressable, different slaves can use
  different SPI modes and speeds (reconfigure between transactions)
- **Disadvantage**: CS pins are limited — an MCU might have only 6 spare GPIOs

**Topology 2: Daisy Chain**

```
  Master                Slave 1                Slave 2
  ┌─────┐   MOSI    ┌─────────┐  MOSI_out  ┌─────────┐
  │     ├───────────→│ MOSI    │───────────→│ MOSI    │
  │     │            │   (shift reg)        │   (shift reg)
  │     │←───────────│ MISO    │←───────────│ MISO    │
  │     │   MISO     └─────────┘            └─────────┘
  │     │
  │ CLK ├───────────→ CLK ──────────────────→ CLK
  │ CS  ├───────────→ CS  ──────────────────→ CS
  └─────┘
```

In daisy-chain mode, **all slaves share a single CS line**. Data flows: master's MOSI →
Slave 1's shift register → Slave 1's MISO → Slave 2's MOSI → Slave 2's shift register
→ Slave 2's MISO → back to master.

To send N bytes to N slaves, the master must clock N×8 bits. The first byte ends up
in the LAST slave (it gets shifted through all prior slaves).

- **Wires needed**: always 4 (CLK, MOSI, MISO, CS) regardless of slave count
- **Advantage**: minimal wiring, scales to many (10+) devices (e.g., LED strip drivers)
- **Disadvantage**: ALL slaves must use the same speed/mode, latency increases linearly,
  every slave must be present (broken chain = no communication)

**Why SPI has no software addressing:**

I2C packs a 7-bit address into the first byte of every transaction. SPI doesn't —
because CS already selects the target. This makes SPI simpler (no address parsing
in slave firmware) but requires dedicated hardware lines. It's a speed-vs-pin-count
tradeoff that SPI chose in favor of speed.

---

### §2 — SPI Clock Modes (CPOL/CPHA) — THE Critical Section

**This section is the single most important part of SPI understanding. A mismatch in
clock mode between master and slave is the #1 cause of SPI bugs.**

---

#### 2.1 CPOL — Clock Polarity (idle state of the SCLK wire)

When no transaction is happening (CS is HIGH, bus is idle), what voltage is the SCLK
pin sitting at?

| CPOL | SCLK idle state | Description |
|------|----------------|-------------|
| 0 | LOW (0V) | "Default" — clock rests at ground, pulses go UP |
| 1 | HIGH (VCC) | Clock rests at VCC, pulses go DOWN |

**Physical intuition**: CPOL=0 means the clock is "normally off" (low), and each bit
gets a positive pulse. CPOL=1 means the clock is "normally on" (high), and each bit
gets a negative pulse.

---

#### 2.2 CPHA — Clock Phase (which edge to SAMPLE data on)

Each clock cycle has TWO edges — a leading edge and a trailing edge. CPHA determines
which one is used to latch (sample) the data bit.

| CPHA | Sample edge | Data valid on |
|------|-------------|--------------|
| 0 | **First** (leading) edge after CS asserts | Data must be READY before first CLK edge |
| 1 | **Second** (trailing) edge | Data setup on first edge, sampled on second |

The "first" and "second" edges depend on CPOL:

| Mode | CPOL | CPHA | Sample Edge | Shift Edge |
|------|------|------|------------|------------|
| 0 | 0 | 0 | Rising ↑ | Falling ↓ |
| 1 | 0 | 1 | Falling ↓ | Rising ↑ |
| 2 | 1 | 0 | Falling ↓ | Rising ↑ |
| 3 | 1 | 1 | Rising ↑ | Falling ↓ |

> **Memory trick**: Modes 0 and 3 both sample on the RISING edge.
> Modes 1 and 2 both sample on the FALLING edge.

---

#### 2.3 Mode 0 (CPOL=0, CPHA=0) — The Most Common Mode

Clock idles LOW. Data is sampled on the RISING edge. Data changes on the FALLING edge.

**The subtlety**: because CPHA=0, the data must be valid BEFORE the first clock edge.
The slave must place bit 7 on MISO when CS goes low, BEFORE any clock pulse.

```
  ┌──── Transaction: Master sends 0xA5, Slave replies 0x3C ────┐

  CS:   ‾‾‾‾\_________________________________________/‾‾‾‾‾
              ↓ data must be ready HERE (before 1st CLK)

  SCLK: ‾‾‾‾‾‾‾_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\__‾‾‾‾‾
              idle  1   2   3   4   5   6   7   8  idle
              LOW  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
                 sample sample ...  (all on rising edges)

  MOSI:  ───╱1╲╱0╲╱1╲╱0╲╱0╲╱1╲╱0╲╱1╲───
             b7  b6  b5  b4  b3  b2  b1  b0
             1    0    1    0    0    1    0    1  = 0xA5

  MISO:  ───╱0╲╱0╲╱1╲╱1╲╱1╲╱1╲╱0╲╱0╲───
             d7  d6  d5  d4  d3  d2  d1  d0
             0    0    1    1    1    1    0    0  = 0x3C

  Legend: ╱x╲ = data bit valid between shift edges
          ↑   = sample point (rising CLK edge)
```

**Used by**: most SPI sensors (ICM-42688, BMI270, BMP388), SD cards (default), many
flash chips.

---

#### 2.4 Mode 1 (CPOL=0, CPHA=1)

Clock idles LOW. Data is sampled on the FALLING edge. Data changes on the RISING edge.

```
  CS:   ‾‾‾‾\_________________________________________/‾‾‾‾‾

  SCLK: ‾‾‾‾‾‾‾_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\__‾‾‾‾‾
                ↑   ↓   ↑   ↓   ↑   ↓   ↑   ↓
               shift sample shift sample  (shift on rise, sample on fall)

  MOSI:  ─────╱ b7╲╱ b6╲╱ b5╲╱ b4╲╱ b3╲╱ b2╲╱ b1╲╱ b0╲──
                    ↓       ↓       ↓       ↓
                  sample points (falling edges)

  MISO:  ─────╱ d7╲╱ d6╲╱ d5╲╱ d4╲╱ d3╲╱ d2╲╱ d1╲╱ d0╲──
```

**Key difference from Mode 0**: data is not required until the FIRST rising edge
(the shift edge). The slave has from CS assertion until the first rising CLK to
prepare its first bit. This gives a half-clock-period more setup time.

**Used by**: some ADCs (MCP3204), certain industrial sensors.

---

#### 2.5 Mode 2 (CPOL=1, CPHA=0)

Clock idles HIGH. Data is sampled on the FALLING edge. Data changes on the RISING edge.

```
  CS:   ‾‾‾‾\_________________________________________/‾‾‾‾‾

  SCLK: ─────‾\‾/\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/───‾‾‾‾
         idle  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   idle
         HIGH sample points (falling edges)           HIGH

  MOSI:  ───╱ b7╲╱ b6╲╱ b5╲╱ b4╲╱ b3╲╱ b2╲╱ b1╲╱ b0╲───

  MISO:  ───╱ d7╲╱ d6╲╱ d5╲╱ d4╲╱ d3╲╱ d2╲╱ d1╲╱ d0╲───
```

**Like Mode 0 but inverted clock**. CPHA=0 still means data must be ready before the
first clock edge. Since the first edge is now FALLING (clock goes from idle HIGH to
LOW), the sample point is FALLING.

**Used by**: some SPI flash chips, certain audio DACs.

---

#### 2.6 Mode 3 (CPOL=1, CPHA=1)

Clock idles HIGH. Data is sampled on the RISING edge. Data changes on the FALLING edge.

```
  CS:   ‾‾‾‾\_________________________________________/‾‾‾‾‾

  SCLK: ─────‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\─────‾‾‾‾
         idle   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   idle
         HIGH  sample points (rising edges)           HIGH

  MOSI:  ─────╱ b7╲╱ b6╲╱ b5╲╱ b4╲╱ b3╲╱ b2╲╱ b1╲╱ b0╲──

  MISO:  ─────╱ d7╲╱ d6╲╱ d5╲╱ d4╲╱ d3╲╱ d2╲╱ d1╲╱ d0╲──
```

**Like Mode 1 but inverted clock.** The slave has extra setup time before the first
sample edge (rising), just like Mode 1.

**Used by**: some SPI slaves in combination with CPOL=1 devices.

---

#### 2.7 The Half-Clock Offset Gotcha — Why CPHA=0 Is Harder on Slaves

This is subtle but critically important for slave-side firmware.

With **CPHA=0**, the first bit of the slave's response must be on MISO **before** the
first clock edge. The timing is:

```
  CPHA=0:
  CS asserts ──→ slave puts bit 7 on MISO ──→ first CLK edge (samples it)
                 must be done in t_CSS      │
                 (CS setup time, 5-50ns)     │

  CPHA=1:
  CS asserts ──→ first CLK edge (shift edge) ──→ slave puts bit 7 ──→ second CLK edge (samples it)
                 nothing sampled yet              has one full half-period to set up
```

At 8 MHz SPI (125ns period, 62.5ns half-period), CPHA=0 gives the slave only t_CSS
(maybe 10ns) to react to CS and put data on MISO. CPHA=1 gives it a full 62.5ns.

**Practical implication**: if your slave has slow interrupt latency or DMA setup time,
CPHA=1 modes (Mode 1 or 3) are more forgiving. Many SPI slave implementations
pre-arm the DMA buffer and hardware handles bit 7 automatically — but if you're
bit-banging a slave, CPHA=0 is much harder to get right.

---

#### 2.8 How to Read a Datasheet to Determine SPI Mode

Datasheets rarely say "use Mode 0." Instead, they show a timing diagram. Here's how
to decode it:

**Step 1: Look at SCLK idle state.**
- If idle LOW → CPOL=0
- If idle HIGH → CPOL=1

**Step 2: Find the sample arrows on the data lines.**
- Is the sample arrow on the FIRST edge after idle? → CPHA=0
- Is the sample arrow on the SECOND edge? → CPHA=1

**Example: ICM-42688-P IMU (InvenSense/TDK)**

From the datasheet timing diagram:
- SCLK idles LOW → CPOL=0
- Data is sampled on the rising edge (first edge from idle LOW) → CPHA=0
- Therefore: **Mode 0**

The datasheet confirms: "Data is sampled on the rising edge of SCLK and driven on
the falling edge."

**Example: AD7606 ADC (Analog Devices)**

From the datasheet:
- SCLK idles HIGH → CPOL=1
- Data sampled on falling edge (first edge from idle HIGH) → CPHA=0
- Therefore: **Mode 2**

> **Warning:** Some datasheets use non-standard terminology. "CPOL" might be called
> "clock polarity," "SPC" (serial port clock), or "CKP." "CPHA" might be "clock phase,"
> "CKE" (Microchip PIC), or "SPI edge select." Always verify by examining the actual
> timing diagram.

---

### §3 — SPI Protocol Details

---

#### 3.1 Full-Duplex Nature — Send AND Receive Simultaneously

**The analogy: a two-lane road**

SPI is like a two-lane road — traffic (data) flows in both directions at the same time.
When the master wants to READ a sensor register, it still must TRANSMIT something on
MOSI (the master-to-slave lane). What it transmits during the read phase is called
"dummy bytes" — typically 0x00 or 0xFF. The slave ignores them.

Conversely, when the master WRITES to a slave, the slave shifts out data on MISO.
Usually this data is meaningless (status bytes, or whatever was left in the shift
register from the last transaction). The master ignores it.

```
  Master sends WRITE command (writes 0x42 to register 0x10):

  Byte 1 (command):    Master sends: 0x10     Slave returns: (garbage)
  Byte 2 (data):       Master sends: 0x42     Slave returns: (garbage)

  Master sends READ command (reads register 0x10):

  Byte 1 (command):    Master sends: 0x90     Slave returns: (garbage)
  Byte 2 (dummy):      Master sends: 0x00     Slave returns: 0xAB ← actual data
                        ↑ dummy byte needed        ↑ THIS is what we want
                          to generate CLK
```

**Why the dummy byte is required**: the SPI clock is generated by the master. Without
clocking MOSI, no CLK edges are produced, and the slave has no way to shift out its
data on MISO. The master must "pump the clock" by sending dummy bytes.

---

#### 3.2 No ACK Mechanism — Silent Failures

**The analogy: mailing a letter with no return receipt**

I2C has a built-in acknowledgement: after every byte, the receiver pulls SDA low to
say "got it." If no ACK comes, the sender knows something is wrong (no device at
that address, device busy, etc.).

SPI has **nothing**. If you:
- Forgot to connect a slave → master reads 0xFF (if MISO has pull-up) or 0x00 (no pull-up)
- Selected wrong CS pin → same as above
- Used wrong SPI mode → reads garbage, no error flag
- Slave crashed mid-transaction → reads garbage or 0xFF

**There is no way for the SPI hardware to distinguish "slave sent 0xFF" from
"no slave connected."**

**Mitigation strategies:**

1. **WHO_AM_I register**: most SPI sensors have a read-only register at a known address
   that returns a fixed value (e.g., ICM-42688 returns 0x47 at register 0x75). Read it
   at startup to verify the device is present and the SPI link works.

2. **CRC in protocol**: if your application protocol includes a CRC (see §6), corrupted
   data will fail the CRC check.

3. **All-0xFF / all-0x00 check**: if you read a register and get 0xFF or 0x00, be suspicious.
   Real sensor data rarely equals these.

---

#### 3.3 Bit Ordering — MSB-First vs LSB-First

Most SPI devices use MSB-first (Most Significant Bit first). The value 0xA5 (binary
10100101) is sent as:

```
  MSB-first: 1, 0, 1, 0, 0, 1, 0, 1  → bit 7 first, bit 0 last
  LSB-first: 1, 0, 1, 0, 0, 1, 0, 1  → bit 0 first, bit 7 last
```

Wait — the bit pattern looks the same! That's because 0xA5 is a palindrome. For a
non-palindrome like 0xC0 (11000000):

```
  MSB-first: 1, 1, 0, 0, 0, 0, 0, 0  ← receiver gets 0xC0  ✓
  LSB-first: 0, 0, 0, 0, 0, 0, 1, 1  ← receiver gets 0x03  ✗ (reversed!)
```

If master sends MSB-first but slave expects LSB-first, every byte is bit-reversed.
0x80 becomes 0x01, 0xFF stays 0xFF, palindromes work fine. This is why the bug
sometimes "seems to work" — some values happen to be correct.

**How to check**: the SPI peripheral register has a bit for LSBFIRST/MSBFIRST. The
device datasheet specifies which one. Zephyr's SPI API uses the `SPI_TRANSFER_LSB` flag.

---

#### 3.4 Word Size

Standard SPI uses 8-bit words, but the hardware supports arbitrary word sizes:

| Word Size | Use Case |
|-----------|----------|
| 4-bit | Some LCD controllers |
| 8-bit | Most devices (sensors, flash, SD cards) |
| 12-bit | Some ADCs (AD7685) |
| 16-bit | Some DACs, audio codecs |
| 32-bit | Some motor controllers, FPGA interfaces |

The SPI shift register must match the word size. If a device uses 16-bit words and
you configure the controller for 8-bit, you'll get byte-swapped data or framing errors.

In Zephyr, word size is set in `struct spi_config`:
```c
struct spi_config cfg = {
    .frequency = 8000000,
    .operation = SPI_OP_MODE_MASTER | SPI_WORD_SET(8),  /* 8-bit words */
};
```

---

#### 3.5 CS Behavior Between Bytes

Some devices require CS to stay LOW for the entire multi-byte transaction (most common).
Others require CS to toggle HIGH between bytes (rare, but exists in some older EEPROMs).

```
  Continuous CS (most devices):

  CS:   ‾\_________________________/‾
  CLK:    _/‾\_/‾\_ ... _/‾\_/‾\_
          byte 1         byte N

  Toggle CS between bytes:

  CS:   ‾\________/‾‾\________/‾
  CLK:    _/‾\_/‾\_     _/‾\_/‾\_
          byte 1         byte 2
```

In Linux spidev, the `cs_change` flag in `struct spi_ioc_transfer` controls this.
In Zephyr, `SPI_HOLD_ON_CS` keeps CS asserted across multiple `spi_transceive` calls
within the same `spi_buf_set`.

---

#### 3.6 Register Access Pattern — The 0x80 Convention

Most SPI sensors follow this convention for register access:

```
  Write register:  TX byte 0 = register_addr       (bit 7 = 0)
                   TX byte 1 = data_to_write

  Read register:   TX byte 0 = 0x80 | register_addr (bit 7 = 1)
                   TX byte 1 = 0x00 (dummy)
                   RX byte 1 = register_value

  Burst read:      TX byte 0 = 0x80 | start_addr
                   TX bytes 1..N = 0x00 (dummies)
                   RX bytes 1..N = register values (auto-increment addr)
```

The `0x80 | addr` pattern sets bit 7 of the address byte, which the slave interprets
as "this is a read, not a write." Not ALL devices use this convention — some use bit 6
for read/write, some use completely different command structures (e.g., SPI flash uses
command opcodes like 0x03 for read, 0x02 for write).

**Always check the datasheet for the specific device's register access protocol.**

---

### §4 — SPI Master vs Slave Operation

---

#### 4.1 Master — The Easy Side

The master has full control:

1. Configure SPI peripheral (mode, speed, word size)
2. Assert CS (pull LOW)
3. Load TX data into shift register or DMA buffer
4. Start clock — hardware shifts data automatically
5. When transfer completes: read RX data, deassert CS

The master decides WHEN to talk, HOW FAST, and to WHOM. There's no urgency —
the master can pause between transactions for as long as it wants.

---

#### 4.2 Slave — The Hard Side

The slave must react to the master's timing. It has NO control over when the
transaction starts or how fast the clock runs.

**The fundamental challenge:**

```
  Timeline:
  ──────────────────────────────────────────────→ time

  CS asserts    First CLK edge     Last CLK edge     CS deasserts
      ↓              ↓                  ↓                ↓
      │← t_CSS →│    │                  │                │
      │  5-50ns  │    │                  │                │
      │          │    │                  │                │
      │ Slave must have MISO bit 7 ready here (CPHA=0)   │
```

The slave must:

1. **Detect CS assertion** — via hardware (SPI peripheral) or GPIO interrupt. Hardware
   detection is preferred because GPIO interrupt latency can be 500ns+.

2. **Have TX data loaded** — the shift register or DMA buffer must contain the response
   data BEFORE the master starts clocking. If the slave firmware needs to calculate
   the response (e.g., read a sensor), it must have done so in advance.

3. **Process RX data after transfer** — once the master clocks the data through, the
   slave reads the received data from the shift register or DMA buffer.

**The critical timing window for CPHA=0:**

Between CS assertion and the first CLK edge, the slave has t_CSS (CS setup time,
often 5-50ns). This is NOT enough time for a CPU interrupt, context switch, and DMA
setup. The slave must either:

a. **Pre-arm DMA** — have the DMA already configured with a buffer BEFORE CS asserts.
   The hardware handles bit shifting automatically.

b. **Use hardware shift register preloading** — some SPI peripherals let you preload a
   value into the TX shift register that gets sent automatically when CS asserts.

---

#### 4.3 The "Always Ready" Pattern for Slaves

In practice, the slave firmware runs in a loop:

```
  1. Prepare TX buffer with current data (or last known good data)
  2. Arm DMA: TX buffer → SPI_DR, SPI_DR → RX buffer
  3. Wait for DMA complete interrupt (triggered by master's CLK)
  4. Process RX buffer (decode master's command)
  5. Prepare new TX buffer based on command
  6. Re-arm DMA → goto 2
```

The critical insight: step 2 (arm DMA) must happen BEFORE the master asserts CS.
This means the slave must have a buffer ready at ALL times. If it's busy calculating
the new response when the master initiates a transaction, the DMA will send the OLD
buffer (stale data). This is acceptable in many protocols — the master reads "last
known" data, and the fresh data appears on the next transaction.

---

#### 4.4 Slave Clock Domain Considerations

The SPI slave's shift register runs on the MASTER's clock, not the slave's system clock.
This creates a clock domain crossing problem:

```
  Master SPI CLK:  8 MHz  ─────→ Slave SPI shift register (8 MHz)
                                            ↕
  Slave system CLK: 32 MHz ─────→ Slave DMA / CPU (32 MHz)
```

The SPI peripheral has synchronization logic to safely transfer data between the 8 MHz
SPI clock domain and the 32 MHz system clock domain. But this has limits:

- **Maximum SPI clock**: typically system_clock / 2 or system_clock / 4 for slave mode.
  If system_clock = 32 MHz, max SPI slave speed might be 8-16 MHz.

- **Metastability**: if SPI CLK and system CLK edges coincide "almost" simultaneously,
  flip-flops can enter a metastable state (output neither 0 nor 1 for a brief time).
  The SPI peripheral's synchronizer (typically a 2-3 flip-flop chain) handles this, but
  adds 2-3 system clock cycles of latency to detect transfers.

---

### §5 — DMA and SPI

---

#### 5.1 Why DMA for SPI

**The analogy: assembly line vs one-at-a-time**

Without DMA, the CPU must intervene for EVERY byte:

```
  Without DMA (interrupt-driven):

  Byte arrives → Interrupt fires → CPU saves context (push registers)
  → ISR reads SPI_DR → ISR writes next byte to SPI_DR → CPU restores context
  → Return from interrupt

  Time per byte: ~20-50 CPU cycles for ISR overhead
  At 8 MHz SPI, 1 byte = 1µs
  64-byte frame = 64 interrupts = 1280-3200 cycles of overhead
```

With DMA:

```
  With DMA:

  DMA armed with: source=tx_buffer, dest=SPI_DR, count=64
  ← DMA hardware copies bytes autonomously. CPU does NOTHING. →
  After 64 bytes: ONE interrupt fires → CPU processes the complete frame

  Time per byte: 0 CPU cycles (DMA hardware handles it)
  Overhead: ONE interrupt for the entire frame
```

**At 8 MHz, the savings are dramatic:**

| Method | Interrupts per 64B frame | CPU time per frame |
|--------|-------------------------|-------------------|
| Polled (busy-wait) | 0 | 64 µs (CPU blocked!) |
| Interrupt per byte | 64 | ~64 µs, but interleaved |
| DMA | 1 | ~1 µs (ISR only) |

DMA frees the CPU to do other work while the SPI transfer proceeds in hardware.

---

#### 5.2 DMA Transfer Anatomy

A DMA transfer is essentially a **hardware-driven memcpy**:

```c
// Software memcpy:
for (int i = 0; i < count; i++) {
    dst[i] = src[i];
}

// DMA "memcpy" (configured via registers):
DMA_SRC_ADDR  = &spi_peripheral->DATA_REG;  // source: SPI data register
DMA_DST_ADDR  = &rx_buffer[0];              // destination: RAM buffer
DMA_COUNT     = 64;                          // transfer 64 units
DMA_SRC_INC   = NO;                          // don't increment source (always read same SPI_DR)
DMA_DST_INC   = YES;                         // increment destination (fill buffer sequentially)
DMA_WIDTH     = BYTE;                        // 8-bit transfers
DMA_TRIGGER   = SPI1_RX_NOT_EMPTY;           // start each transfer when SPI has a byte
DMA_ENABLE    = 1;                           // go!
```

Once enabled, the DMA controller watches for the trigger event (SPI has received a
byte). When triggered, it reads one byte from `SPI_DR`, writes it to `rx_buffer[i]`,
increments `i`, and decrements the count. When count reaches 0, it fires an interrupt.

**TX DMA works in reverse:**

```
DMA_SRC_ADDR  = &tx_buffer[0];              // source: RAM buffer
DMA_DST_ADDR  = &spi_peripheral->DATA_REG;  // destination: SPI data register
DMA_SRC_INC   = YES;                         // read successive bytes from buffer
DMA_DST_INC   = NO;                          // always write to same SPI_DR
DMA_TRIGGER   = SPI1_TX_EMPTY;               // feed SPI when it's ready for next byte
```

---

#### 5.3 DMA Channels and Streams — Hardware Mapping

On STM32, the DMA controller has a fixed mapping from peripherals to DMA channels:

```
  STM32H743 DMA1 mapping (simplified):

  Stream 0: SPI3_RX or I2C1_RX
  Stream 1: SPI3_TX
  Stream 2: SPI1_RX
  Stream 3: SPI1_TX or I2C1_TX   ← CONFLICT!
  Stream 4: SPI2_RX
  Stream 5: SPI2_TX
  ...
```

**The conflict problem**: Stream 3 is shared between SPI1_TX and I2C1_TX. If your
design uses both SPI1 and I2C1, they cannot both transmit via DMA simultaneously.
You must either:
- Time-multiplex (use SPI1 DMA when I2C1 is idle, and vice versa)
- Use DMA2 for one peripheral (STM32H7 has two DMA controllers)
- Use BDMA (Basic DMA) for simpler peripherals
- Fall back to interrupt-driven for the lower-priority peripheral

**Zephyr handles this in devicetree:**
```dts
&spi1 {
    dmas = <&dma1 2 0x00000440>, <&dma1 3 0x00000440>;
    dma-names = "rx", "tx";
};
```

If there's a conflict, the build will succeed but the runtime behavior is undefined
(both peripherals fight for the same DMA stream). You must check the reference manual's
DMA request mapping table during hardware design.

---

#### 5.4 Circular vs Normal DMA Mode

| Mode | Behavior | Use Case |
|------|----------|----------|
| Normal | Transfer `count` items, then STOP. Fire complete interrupt. | SPI frames (finite length) |
| Circular | Transfer `count` items, RESET counter, start over from beginning. | Continuous ADC sampling, audio streams |

For SPI, **Normal mode** is almost always correct. A SPI transaction has a defined
length — you want exactly 64 bytes transferred, then stop.

Circular mode would re-arm automatically and overwrite the buffer with the next
transaction's data before you've processed the first one. Dangerous.

**Exception**: circular mode is useful for SPI audio interfaces (I2S) where data
streams continuously and you process it in chunks using half-complete and complete
interrupts (double-buffering pattern).

---

#### 5.5 Double-Buffering with DMA — The Ping-Pong Pattern

When transaction rates are high (>1 kHz) and you need zero-gap processing:

```
  Buffer A ──── DMA ──── SPI
                          ↕
  Buffer B ──── CPU processes previous frame

  Time ─────────────────────────────────────→

  Phase 1: DMA fills Buffer A       CPU processes Buffer B (from last transaction)
  Phase 2: (DMA complete interrupt) → swap: DMA fills Buffer B, CPU processes Buffer A
  Phase 3: (DMA complete interrupt) → swap again: DMA fills Buffer A, CPU processes Buffer B
  ...
```

**State machine:**

```
                ┌──────────────────────────────┐
                │                              │
                ▼                              │
  ┌──── ARM DMA (Buffer A) ────┐               │
  │                            │               │
  │                 DMA complete IRQ            │
  │                            │               │
  │                            ▼               │
  │          ┌── SWAP ──→ ARM DMA (Buffer B)   │
  │          │               │                 │
  │    CPU processes A       │ DMA complete    │
  │          │               │ IRQ             │
  │          │               ▼                 │
  │          └───── SWAP ──→ ARM DMA (Buffer A)│
  │                          │                 │
  │                    CPU processes B         │
  │                          │                 │
  └──────────────────────────┘─────────────────┘
```

On STM32H7, the DMA controller has a native double-buffer mode: two memory addresses
configured, and the hardware swaps automatically at count=0. Zephyr doesn't directly
expose this — you implement it with two DMA calls and a flag.

---

#### 5.6 Cache Coherency with DMA — The D-Cache Trap

**This is the #1 DMA bug on Cortex-M7 (STM32H7, i.MX RT1060, etc.).**

Cortex-M7 has a Data Cache (D-Cache) between the CPU and SRAM. It makes CPU memory
access faster by keeping recently-used data in fast cache memory. But DMA does NOT
go through the cache — it reads/writes directly to SRAM.

```
  CPU ←→ D-Cache ←→ SRAM ←→ DMA ←→ SPI peripheral

  The problem:

  1. CPU writes tx_buffer[0] = 0x42
     → This write goes to D-Cache, NOT to SRAM yet (write-back policy)
     → SRAM still has old data (say, 0x00)

  2. DMA reads from SRAM address of tx_buffer[0]
     → DMA gets 0x00 (stale SRAM data), NOT 0x42 (cache has it but DMA can't see it)
     → WRONG DATA SENT OVER SPI

  Fix: clean (flush) the cache before starting TX DMA:
     SCB_CleanDCache_by_Addr(tx_buffer, sizeof(tx_buffer));
     → Forces cached data to SRAM → DMA gets 0x42 ✓
```

The reverse problem on RX:

```
  1. DMA writes received data to SRAM: rx_buffer[0] = 0xAB
  2. CPU reads rx_buffer[0]
     → CPU reads from D-Cache, which still has old cached value (say, 0x00)
     → CPU gets 0x00, NOT 0xAB
     → WRONG DATA RETURNED TO APPLICATION

  Fix: invalidate the cache after RX DMA completes:
     SCB_InvalidateDCache_by_Addr(rx_buffer, sizeof(rx_buffer));
     → Cache entries marked invalid → next CPU read goes to SRAM → gets 0xAB ✓
```

**The complete DMA-safe SPI transaction:**

```c
// 1. Prepare TX buffer
memcpy(tx_buf, data, len);

// 2. Clean cache: push CPU writes to SRAM so DMA can see them
SCB_CleanDCache_by_Addr((uint32_t*)tx_buf, len);

// 3. Start DMA transfer (TX and RX simultaneously)
HAL_SPI_TransmitReceive_DMA(&hspi1, tx_buf, rx_buf, len);

// 4. Wait for DMA complete (ISR sets flag)
while (!dma_complete_flag) { /* or use semaphore */ }

// 5. Invalidate cache: discard CPU's stale cache of the RX region
SCB_InvalidateDCache_by_Addr((uint32_t*)rx_buf, len);

// 6. Now rx_buf[] is safe to read
```

**Buffer alignment requirement**: `SCB_CleanDCache_by_Addr` operates on 32-byte cache
lines. If your buffer isn't aligned to 32 bytes, the clean/invalidate operation
affects neighboring memory too, potentially corrupting other data. Always:
```c
__attribute__((aligned(32))) uint8_t tx_buf[64];
__attribute__((aligned(32))) uint8_t rx_buf[64];
```

Also ensure buffer sizes are multiples of 32 bytes (pad if needed).

---

### §6 — SPI Framing and Error Detection

---

#### 6.1 The Framing Problem

SPI has NO built-in framing. Unlike UART (which has start/stop bits that define byte
boundaries), SPI is just a continuous bitstream clocked by CLK. If synchronization is
lost — e.g., EMI causes a spurious clock pulse, or the master and slave disagree on
word size — every subsequent byte is shifted by one or more bits.

```
  Normal:    |b7 b6 b5 b4 b3 b2 b1 b0|b7 b6 b5 b4 b3 b2 b1 b0|
             |←── byte 0 ────────────→|←── byte 1 ────────────→|

  After spurious extra CLK pulse (1 bit lost):

             |b6 b5 b4 b3 b2 b1 b0 b7|b6 b5 b4 b3 b2 b1 b0 b7|
             |←── byte 0 (WRONG!) ───→|←── byte 1 (WRONG!) ───→|

  Every byte is now shifted left by 1 bit. The MSB of byte N
  becomes the LSB of byte N-1. All data is garbage.
```

**Detection**: toggling CS resets the slave's shift register. Every transaction starts
with a fresh CS assert. If you keep CS LOW for very long multi-byte transfers, the
risk of sync loss increases.

---

#### 6.2 Magic Byte / Header

The simplest framing method: start every frame with a known constant byte.

```
  Frame:  [0xAA] [length] [payload...] [CRC]
           ↑ magic byte: receiver scans for this to find frame start
```

**Problem**: 0xAA can appear naturally in the payload. False sync recovery. Unless
combined with a CRC that validates the entire frame, magic-byte-only framing is fragile.

---

#### 6.3 CRC — Cyclic Redundancy Check

CRC treats the data as a polynomial and divides it by a fixed "generator polynomial."
The remainder is the CRC value. The receiver performs the same division — if the
remainder doesn't match, the data was corrupted.

**CRC-8 example (polynomial 0x07, aka CRC-8/CCITT):**

For the data bytes `[0x03, 0x14]`:

```
  Step 1: Append 8 zero bits (space for CRC)
  Data:   0000_0011  0001_0100  0000_0000
          ↑ 0x03     ↑ 0x14    ↑ placeholder for CRC

  Step 2: Polynomial long division (XOR instead of subtraction)
  Generator polynomial: 0x07 = 0000_0111

  (Division proceeds bit by bit — detailed steps omitted for brevity)

  Step 3: Remainder = CRC value

  Frame sent: [0x03, 0x14, CRC]
```

**Detection capabilities of CRC-8:**
- All single-bit errors: detected 100%
- All odd numbers of bit errors: detected 100%
- Burst errors ≤8 bits: detected 100%
- Random longer errors: detected 99.6%

**CRC-16/CCITT** (polynomial 0x1021) is stronger — detects all bursts ≤16 bits.
Commonly used in SPI protocols with reliability requirements.

Most MCUs have a hardware CRC peripheral that computes CRC in 1-2 clock cycles per
byte. Much faster than software CRC.

---

#### 6.4 Length-Prefixed Framing

The most robust approach for custom SPI protocols:

```
  Frame format:
  ┌──────┬──────┬────────────────┬──────┐
  │ HDR  │ LEN  │   PAYLOAD      │ CRC  │
  │ 0xAA │ N    │ N bytes        │ 2B   │
  └──────┴──────┴────────────────┴──────┘
  1 byte  1 byte  variable        CRC16

  HDR:     Magic byte for frame start detection
  LEN:     Payload length (max 255 bytes)
  PAYLOAD: Application data
  CRC:     CRC-16 over HDR + LEN + PAYLOAD
```

The receiver knows exactly how many bytes to expect (LEN), and CRC validates the
entire frame. If LEN or CRC is corrupted, the receiver discards the frame and
re-synchronizes on the next 0xAA.

---

#### 6.5 Sequence Numbers

To detect dropped or duplicated frames, add a sequence number:

```
  Frame: [HDR] [SEQ] [LEN] [PAYLOAD] [CRC]
               ↑ 8-bit counter: 0, 1, 2, ..., 255, 0, 1, ...
```

The receiver tracks the expected sequence number. If it receives SEQ=5 after SEQ=3,
it knows one frame was lost. If it receives SEQ=3 twice, it knows a frame was
duplicated (or the sender retransmitted).

**In the OKS sensorbar protocol**: sequence numbers revealed that sensorbar frames
were being duplicated at the firmware level — the same SEQ appeared twice with
the same data, indicating the firmware was reading stale data from the SPI buffer.

---

### §7 — Failure Modes and Debugging

---

#### 7.1 Wrong Mode (CPOL/CPHA Mismatch)

**Symptom**: data looks "almost right" — some bits correct, some flipped. Or completely
random garbage.

**Why**: if the master samples on the rising edge (Mode 0) but the slave shifts data
on the rising edge (Mode 3 from slave's perspective), the master captures data while
it's transitioning — the electrical level is indeterminate.

**Diagnosis with logic analyzer**:
1. Capture CLK, MOSI, MISO, CS simultaneously
2. Look at which CLK edge the data transitions on
3. Look at which CLK edge the data should be sampled on
4. If they're the same edge → mode mismatch (data changes AT the sample point)

**Fix**: change the master's SPI mode to match the slave's datasheet specification.

---

#### 7.2 CS Timing Violation

**Symptom**: first byte of every transaction is garbage, remaining bytes are correct.

**Why**: the master asserts CS and starts CLK too quickly. The slave hasn't finished
configuring its SPI peripheral or loading the TX shift register.

**Diagnosis**: measure CS-to-first-CLK time on oscilloscope. Compare to datasheet's
t_CSS minimum. If it's violated, add a delay:

```c
gpio_pin_set(cs_port, cs_pin, 0);   // assert CS
k_busy_wait(1);                      // 1 µs delay (generous)
spi_transceive(...);                 // start CLK
```

---

#### 7.3 Signal Integrity Failure

**Symptom**: works at 1 MHz, fails at 8 MHz. Or works on short wires, fails on longer ones.

**Why**: ringing on CLK causes the receiver to see extra clock edges (double-clocking).
A logic analyzer (which has input thresholds and hysteresis) may show clean edges
while an oscilloscope shows the true analog waveform with ringing.

**Diagnosis**:
1. Use an OSCILLOSCOPE (not logic analyzer) on CLK at the slave's pin
2. Look for ringing that crosses the input threshold
3. Reduce SPI speed until it works, then add series termination resistors

---

#### 7.4 MISO Contention

**Symptom**: data corruption when multiple slaves are on the bus, even though only one
is CS-asserted.

**Why**: a slave with a firmware bug isn't tristating its MISO when CS is deasserted.
Both the selected slave AND the buggy slave drive MISO simultaneously.

**Diagnosis**:
1. Disconnect all slaves except the target → does it work? → another slave is fighting
2. Measure MISO voltage when the target slave is CS-selected → if voltage is ~VCC/2
   (e.g., 1.6V for 3.3V bus) instead of clean HIGH/LOW → two drivers fighting

---

#### 7.5 DMA Cache Bugs (Cortex-M7 Specific)

**Symptom**: data sent over SPI doesn't match what's in the TX buffer when inspected
with GDB. Or received data buffer shows old values even though the scope shows correct
data on the wire.

**Why**: D-Cache coherency issue (see §5.6). GDB reads through the CPU, which sees
the cached value. DMA bypasses the cache.

**Diagnosis**: add `SCB_CleanDCache_by_Addr` before TX DMA and
`SCB_InvalidateDCache_by_Addr` after RX DMA. If the problem disappears, it was cache
coherency.

**Alternative**: place DMA buffers in non-cacheable SRAM (SRAM4 on STM32H7):
```c
__attribute__((section(".sram4"))) uint8_t tx_buf[64];
```
This avoids cache issues entirely but is slower for CPU access.

---

#### 7.6 Endianness Mismatch

**Symptom**: 16-bit or 32-bit values are byte-swapped. A sensor reporting 0x0180 (384
decimal) is read as 0x8001 (32769 decimal).

**Why**: ARM Cortex-M is little-endian (LSB at lowest address). SPI typically sends
MSB first. A 16-bit value 0x0180:
- In memory (little-endian): address N = 0x80, address N+1 = 0x01
- Over SPI (MSB first): first byte = 0x01, second byte = 0x80

If you `memcpy` from the SPI RX buffer to a `uint16_t`, the byte order depends on
your architecture. On ARM:
```c
uint16_t value;
memcpy(&value, &rx_buf[0], 2);
// value = 0x8001 on little-endian ARM (rx_buf[0]=0x01 goes to LSB)
// Expected: 0x0180
```

**Fix**: use `be16toh()` / byte-swap macros, or manually construct the value:
```c
uint16_t value = ((uint16_t)rx_buf[0] << 8) | rx_buf[1];
```

---

## PART 2 — Gotcha Table

| # | Gotcha | What Goes Wrong | How to Prevent |
|---|--------|----------------|----------------|
| 1 | Wrong SPI mode | Data shifted by 1 bit or random | Check datasheet timing diagram, verify CPOL/CPHA |
| 2 | CS timing too fast | First byte garbage | Add delay between CS assert and first CLK |
| 3 | MISO contention | Voltage at VCC/2, garbage data | Verify all non-selected slaves tristate MISO |
| 4 | Level mismatch (5V ↔ 3.3V) | Overvoltage damage or misread bits | Use level shifter or 5V-tolerant GPIOs |
| 5 | DMA cache stale TX | Wrong data sent, GDB shows correct buffer | SCB_CleanDCache before TX DMA |
| 6 | DMA cache stale RX | Old data in buffer after receive | SCB_InvalidateDCache after RX DMA |
| 7 | DMA buffer unaligned | Cache clean/invalidate corrupts neighbors | 32-byte align + pad DMA buffers |
| 8 | Endianness mismatch | 16/32-bit values byte-swapped | Manual byte-swap or be16toh()/be32toh() |
| 9 | LSB vs MSB first | All values bit-reversed | Check SPI_TRANSFER_LSB flag matches device |
| 10 | DMA channel conflict | Two peripherals fight for same DMA stream | Check reference manual DMA mapping table |
| 11 | CS stays low between transactions | Slave doesn't reset state machine | Ensure CS goes HIGH between separate commands |
| 12 | Wrong word size | Frame shifted, CRC fails | Match SPI controller word size to device requirement |
| 13 | Pull-up on MISO | Disconnected slave reads as 0xFF, not error | Check WHO_AM_I register at startup |
| 14 | Baudrate too high for slave mode | Bit errors at high speed | Slave max SPI CLK ≤ system_clock/2 |
| 15 | Ringing causes double-clocking | Intermittent bit errors at high speed | Add 33Ω series resistors on CLK, check with scope |

---

## PART 3 — Quick Reference Card

---

### SPI Mode Table

| Mode | CPOL | CPHA | CLK Idle | Sample Edge | Shift Edge | Common Devices |
|------|------|------|----------|-------------|------------|----------------|
| 0 | 0 | 0 | LOW | Rising ↑ | Falling ↓ | ICM-42688, BMI270, SD card, most sensors |
| 1 | 0 | 1 | LOW | Falling ↓ | Rising ↑ | MCP3204 ADC, some EEPROMs |
| 2 | 1 | 0 | HIGH | Falling ↓ | Rising ↑ | AD7606, some SPI flash |
| 3 | 1 | 1 | HIGH | Rising ↑ | Falling ↓ | MAX31855, some audio DACs |

---

### Common SPI Speeds by Peripheral Type

| Peripheral | Typical Speed | Max Speed | Notes |
|-----------|--------------|-----------|-------|
| SD card (SPI mode) | 400 kHz init, 25 MHz data | 50 MHz | Init at 400 kHz, then speed up |
| IMU (ICM-42688) | 1 MHz config, 24 MHz data | 24 MHz | Fast for sensor reads |
| SPI Flash (W25Q128) | 8-50 MHz | 133 MHz (quad) | Read faster than write |
| Display (ILI9341) | 10-40 MHz | 62.5 MHz | Pixel data is bandwidth-hungry |
| ADC (AD7606) | 1-20 MHz | 23 MHz | Speed depends on resolution mode |
| DAC (MCP4922) | 8-20 MHz | 20 MHz | |
| Sensorbar (OKS) | 8 MHz | 8 MHz | Custom protocol, 64-byte frames |

---

### Wire Connection Table (Master ↔ Slave)

| Master Pin | Slave Pin | Direction | Function |
|-----------|-----------|-----------|----------|
| MOSI | MOSI (or SDI, DIN, SI) | Master → Slave | Master Out, Slave In |
| MISO | MISO (or SDO, DOUT, SO) | Slave → Master | Master In, Slave Out |
| SCLK | SCLK (or SCK, CLK) | Master → Slave | Clock (master drives) |
| CS | CS (or SS, NSS, CSN, CE) | Master → Slave | Chip Select (active LOW) |
| GND | GND | Shared | Common ground reference |

> **Note**: GND connection is CRITICAL. SPI relies on voltage levels relative to ground.
> Missing GND = random data. Always connect GND first.

---

### DMA Checklist

```
Before enabling SPI DMA:

□ DMA channel matches SPI peripheral (check reference manual table)
□ No DMA channel conflicts with other peripherals
□ Buffer aligned to 32 bytes (Cortex-M7 with D-Cache)
□ Buffer size is multiple of 32 bytes
□ TX: SCB_CleanDCache_by_Addr() called before starting DMA
□ RX: SCB_InvalidateDCache_by_Addr() called after DMA complete
□ DMA mode is Normal (not Circular) for framed SPI
□ Transfer complete interrupt enabled and ISR registered
□ Source/Dest increment configured correctly:
    TX: src=INC, dst=NO_INC (buffer→SPI_DR)
    RX: src=NO_INC, dst=INC (SPI_DR→buffer)
□ Transfer width matches word size (BYTE for 8-bit SPI)
```

---

### SPI Debugging Flowchart

```
  SPI not working?
      │
      ├─ All bytes = 0xFF or 0x00?
      │   └─ YES → No slave responding
      │        ├─ Check CS pin (is it actually toggling?)
      │        ├─ Check wiring (MOSI↔MOSI, not MOSI↔MISO crossover?)
      │        ├─ Check power (is slave powered? decoupling caps?)
      │        └─ Read WHO_AM_I register → wrong value? → wrong SPI mode
      │
      ├─ First byte wrong, rest OK?
      │   └─ CS timing violation → add delay after CS assert
      │
      ├─ Every byte wrong but consistent?
      │   ├─ Bit-shifted → wrong SPI mode (CPOL/CPHA)
      │   ├─ Bit-reversed → LSB/MSB mismatch
      │   └─ Byte-swapped (16-bit values) → endianness issue
      │
      ├─ Works at low speed, fails at high speed?
      │   └─ Signal integrity → oscilloscope on CLK at slave pin
      │       └─ Ringing? → add 33Ω series termination
      │
      ├─ Works in GDB, fails at full speed?
      │   └─ DMA cache coherency issue (Cortex-M7)
      │       ├─ Clean cache before TX DMA
      │       └─ Invalidate cache after RX DMA
      │
      └─ Intermittent corruption?
          ├─ MISO contention (disconnect other slaves to test)
          ├─ EMI from nearby switching regulator
          └─ Ground loop (check GND connectivity)
```

---

## PART 4 — Annotated Diagrams

---

### Full SPI Transaction (Mode 0, 8-bit, Read Register)

```
  Master reads register 0x75 (WHO_AM_I) from ICM-42688. Expected return: 0x47.

  Step 1: Master sends 0xF5 (0x80 | 0x75 = read flag + register address)
  Step 2: Master sends 0x00 (dummy byte) to clock out slave's response

  CS:   ‾‾‾‾‾‾\________________________________________________/‾‾‾‾‾
                ← entire transaction under one CS assertion →

  SCLK: ‾‾‾‾‾‾‾_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\__/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\__
                |←──────── byte 1 (8 clocks) ──────→|←──── byte 2 (8 clocks) ────→|

  MOSI:        |1  1  1  1  0  1  0  1|0  0  0  0  0  0  0  0|
               |←── 0xF5 (read reg 0x75) ──→|←── 0x00 (dummy) ────→|

  MISO:        |x  x  x  x  x  x  x  x|0  1  0  0  0  1  1  1|
               |←── garbage (slave decoding cmd) →|←── 0x47 (WHO_AM_I) ──→|
                                                   ↑ THIS is the actual data

  Total: 16 CLK cycles, 2 bytes TX, 2 bytes RX (byte 2 of RX is the useful data)
```

---

### DMA-SPI Data Flow (STM32)

```
  ┌─────────────────────────────────────────────────────┐
  │                    CPU / Code                       │
  │                                                     │
  │  tx_buf[] ──write──→ D-Cache ──clean──→ SRAM        │
  │                                          │          │
  │                                     DMA reads from  │
  │                                     SRAM directly   │
  │                                          │          │
  │                                          ▼          │
  │                        ┌────────────────────┐       │
  │                        │   DMA Controller   │       │
  │                        │   Stream/Channel   │       │
  │                        └──────┬─────────────┘       │
  │                               │                     │
  │                               ▼                     │
  │                        ┌────────────┐               │
  │                        │ SPI_DR     │ → CLK/MOSI →  │
  │                        │ (data reg) │ ← MISO ←      │
  │                        └────────────┘               │
  │                               │                     │
  │                        DMA writes to                │
  │                        SRAM directly                │
  │                               │                     │
  │  rx_buf[] ←──read──── D-Cache ←─invalidate── SRAM  │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  Critical cache operations:
  ① SCB_CleanDCache_by_Addr(tx_buf) — before starting DMA TX
  ② SCB_InvalidateDCache_by_Addr(rx_buf) — after DMA RX complete
```

---

*End of study notes. Proceed to exercises for hands-on practice.*
