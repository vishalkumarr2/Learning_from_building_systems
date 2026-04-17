# 06 — I2C Deep Dive
### From open-drain physics to complete protocol transactions
**Prerequisites:** Chapter 01 (RC time constants, pull-ups), Chapter 02 (MOSFETs, open-drain)
**Unlocks:** EEPROM access, sensor configuration (IMU, temp, pressure), IO expanders, multi-sensor buses

---

## Why Should I Care? (STM32/Jetson Project Context)

- The ICM-42688 IMU supports I2C (in addition to SPI) — many breakout boards default to I2C
- I2C EEPROMs store calibration data and serial numbers on production boards
- Temperature sensors (TMP102), IO expanders (PCA9555), and other peripherals sit on I2C
- When I2C misbehaves ("bus stuck"), you need to understand the physics to recover
- Understanding I2C's open-drain nature is essential for designing multi-sensor harnesses on robots

---

# PART 1 — THE PHYSICS: OPEN-DRAIN AND PULL-UPS

---

## 1.1 What "Open-Drain" Means — The Fundamental Insight

Every I2C output stage is an N-channel MOSFET (or BJT) that can only PULL the line LOW. It CANNOT drive the line HIGH. This is called "open-drain" because the MOSFET drain is left "open" (not connected to VCC).

```
    Inside every I2C device's output pin:

    SDA or SCL bus wire
         │
         │ ← this point is either:
         │    - PULLED LOW by the MOSFET (device writes "0")
         │    - LEFT FLOATING (device writes "1" → pull-up makes it HIGH)
         │
        ┌┤ N-channel MOSFET
   G ──┤│
        │
       GND

    When Gate = HIGH: MOSFET ON → wire pulled to GND (logic 0)
    When Gate = LOW:  MOSFET OFF → wire released → pull-up resistor pulls to VCC (logic 1)
```

**Why not just drive HIGH?**

If two devices tried to drive the same wire — one HIGH (VCC) and one LOW (GND) — you'd get a short circuit (VCC directly shorted to GND through the two output stages). Smoke. Dead chips.

With open-drain, this can't happen:
- Device A pulls LOW (MOSFET on) + Device B releases (MOSFET off) → wire is LOW ✓
- Device A releases + Device B pulls LOW → wire is LOW ✓
- Device A pulls LOW + Device B pulls LOW → wire is LOW ✓ (both MOSFETs to GND, no conflict)
- Device A releases + Device B releases → pull-up pulls HIGH ✓

**This is called "wired-AND"**: the line is HIGH only if ALL devices release it. Any single device can pull it LOW. This is safe for multi-device buses.

```
    WIRED-AND truth table:

    Device A  Device B  Bus Level   Logic
    Release   Release   HIGH (1)    1 AND 1 = 1
    Release   Pull LOW  LOW (0)     1 AND 0 = 0
    Pull LOW  Release   LOW (0)     0 AND 1 = 0
    Pull LOW  Pull LOW  LOW (0)     0 AND 0 = 0

    → ANY device pulling LOW makes the bus LOW
    → Bus is HIGH only when ALL devices release
```

---

## 1.2 The Pull-Up Resistor — Why Its Value Matters

The pull-up is what makes the wire go HIGH. It's an external resistor between the bus wire and VCC.

```
    VCC (3.3V)
      │
     ┌┴┐
     │Rp│  ← Pull-up resistor
     └┬┘
      │
      ├── SDA bus wire ── Device 1 ── Device 2 ── Device 3
      │
     ┌┴┐
     │Rp│  ← Pull-up (separate for SCL, same value)
     └┬┘
      │
      ├── SCL bus wire ── Device 1 ── Device 2 ── Device 3
```

**The RC time constant problem:**

The bus wire has capacitance: each device adds ~10pF for its input pin, plus ~50pF per 10cm of wire. The pull-up must charge this capacitance to VCC quickly enough for the bus speed.

```
    Rising edge is a capacitor charging through a resistor:

    V(t) = VCC × (1 - e^(-t/(Rp × Cbus)))

    The I2C spec requires:
    - Standard mode (100kHz): rise time < 1000ns
    - Fast mode (400kHz): rise time < 300ns
    - Fast mode+ (1MHz): rise time < 120ns

    Rise time ≈ 0.8473 × Rp × Cbus  (time to reach 0.7×VCC from 0.3×VCC)
```

### Worked Example: Pull-Up for 400kHz I2C

**Given:**
- 3 devices on the bus, each with 10pF input capacitance = 30pF
- 15cm of wire at ~50pF/10cm = 75pF
- Total Cbus = 30pF + 75pF = 105pF ≈ say 200pF with safety margin
- Required rise time < 300ns (Fast mode)

**Calculate maximum Rp:**

```
    300ns > 0.8473 × Rp × 200pF
    Rp < 300ns / (0.8473 × 200×10⁻¹²)
    Rp < 300×10⁻⁹ / 169.5×10⁻¹²
    Rp < 1770Ω
```

**Calculate minimum Rp (limited by sink current):**

The I2C spec says devices must sink at least 3mA (standard) or 20mA (fast-mode plus). When a device pulls the line LOW to near 0V, the pull-up supplies:

```
    I_sink = VCC / Rp = 3.3V / Rp

    For 3mA max sink: Rp > 3.3V / 3mA = 1100Ω
    For 20mA max sink (Fm+): Rp > 3.3V / 20mA = 165Ω
```

**Result for 400kHz, 200pF bus, 3.3V, 3mA sink:**

```
    1100Ω < Rp < 1770Ω

    Common values in this range: 1.5kΩ or 1.8kΩ
    
    The classic "4.7kΩ I2C pull-up" is for 100kHz mode with low bus capacitance.
    For 400kHz, you likely need 1.5kΩ–2.2kΩ!
```

**Rule of thumb:**
| I2C Speed | Typical Pull-Up (3.3V, <200pF) |
|----------|-------------------------------|
| 100kHz (Standard) | 4.7kΩ – 10kΩ |
| 400kHz (Fast) | 1.5kΩ – 4.7kΩ |
| 1MHz (Fast+) | 1kΩ – 2.2kΩ |

---

# PART 2 — I2C BIT-LEVEL PROTOCOL

---

## 2.1 START and STOP Conditions

I2C defines special conditions using the relationship between SDA and SCL:

```
    START condition: SDA goes LOW while SCL is HIGH
    STOP condition:  SDA goes HIGH while SCL is HIGH
    
    During normal data: SDA only changes while SCL is LOW

    START:                              STOP:
    
    SDA: ‾‾‾‾‾\________                SDA: _________/‾‾‾‾‾
                                        
    SCL: ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                SCL: ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
              ↑                                    ↑
         SDA falls while SCL HIGH            SDA rises while SCL HIGH
         = START                             = STOP
```

**Why these are special:** During normal data transfer, SDA only changes when SCL is LOW (setup time). SDA changing while SCL is HIGH is FORBIDDEN during data — and this is exactly what makes START/STOP unambiguous markers.

---

## 2.2 Data Transfer — One Bit at a Time

```
    Rules:
    1. SDA must be STABLE while SCL is HIGH (this is when the receiver samples)
    2. SDA can CHANGE while SCL is LOW (this is when the transmitter sets up the next bit)
    3. MSB is transmitted first (opposite of UART!)

    One bit:
    
    SCL: ___|‾‾‾‾‾|___
              ↑
         Data sampled here
         (SDA must be stable)
    
    SDA: ===|XXXXX|===
         ↑           ↑
    Changes OK    Changes OK
    (SCL is LOW)  (SCL is LOW)
```

---

## 2.3 Byte Transfer — 8 Data + 1 ACK = 9 Clock Cycles

After every 8 data bits, the receiver sends an ACK (acknowledge) or NACK (not-acknowledge):

```
    Transmitter sends 8 bits (MSB first), then RELEASES SDA:

    SCL: _|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_
          D7  D6  D5  D4  D3  D2  D1  D0  ACK
          ←────── 8 data bits ──────→    ←───→
          
    ACK = 0 (receiver pulls SDA LOW): "Got it, send more"
    NACK = 1 (receiver releases SDA, stays HIGH): "Stop" or "error"
```

---

## 2.4 The Address Byte — 7-bit Address + R/W Bit

Every I2C transaction starts with a START condition followed by the address byte:

```
    Byte structure: [A6][A5][A4][A3][A2][A1][A0][R/W̄]

    A6:A0 = 7-bit slave address
    R/W̄ = 0 for WRITE, 1 for READ
```

**Example: ICM-42688 at address 0x68**

```
    Write to 0x68:
    Binary: [1][1][0][1][0][0][0][0] = 0xD0        (0x68 << 1 | 0)
    
    Read from 0x68:
    Binary: [1][1][0][1][0][0][0][1] = 0xD1        (0x68 << 1 | 1)
```

**Confusion alert:** Datasheets sometimes specify the I2C address as:
- **7-bit format:** 0x68 (you shift left and add R/W in your code)
- **8-bit format (write):** 0xD0 (already shifted, 0 for write)
- **8-bit format (read):** 0xD1 (already shifted, 1 for read)

Check which format the datasheet uses. Getting this wrong means you're addressing the wrong device.

---

## 2.5 Complete Transaction Walkthrough — Reading WHO_AM_I from ICM-42688

**Goal:** Read register 0x75 (WHO_AM_I) from device at address 0x68. Expected return value: 0x47.

**Two-phase transaction:**
1. WRITE phase: tell the device which register you want
2. READ phase: read the register value back

```
    Phase 1 — WRITE register address:
    
    [START] [0xD0] [ACK] [0x75] [ACK]
     ↑       ↑      ↑     ↑      ↑
     |    Address    |  Register  |
     |    0x68+W     |   0x75     |
     |    = 0xD0     | (WHO_AM_I) |
     |               |            |
     SDA LOW while   Slave sends  Slave sends
     SCL HIGH        ACK (SDA=0)  ACK (SDA=0)

    Phase 2 — Repeated START + READ:
    
    [Sr] [0xD1] [ACK] [0x47] [NACK] [STOP]
     ↑     ↑      ↑     ↑      ↑      ↑
     |  Address    |  Data from  |    SDA HIGH
     |  0x68+R     |   slave     |    while SCL
     |  = 0xD1     |  (0x47!)    |    HIGH
     |             |             |
    Repeated      Slave ACKs   Master sends
    START         its addr     NACK → "I don't
    (no STOP                   want more bytes"
     in between!)              then STOP
```

**Detailed bit-by-bit waveform for Phase 1 (writing 0xD0 then 0x75):**

```
    SDA: ‾‾\_1_/‾1‾\_0_/‾1‾\_0_/‾0‾\_0_/‾0‾\_A_/‾0‾\_1_/‾1‾\_1_/‾0‾\_1_/‾0‾\_1_/‾A‾/
              D7      D6      D5      D4      D3      D2      D1      D0     ACK
              ←──────────── 0xD0 (address + write) ────────────→  ← slave pulls LOW

    SCL: ___/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/‾‾\_/...
           ↑START

    Then 0x75 = 0111_0101:
    SDA: \_0_/‾1‾/‾1‾/‾1‾\_0_/‾1‾\_0_/‾1‾\_A_/
          D7   D6   D5   D4   D3   D2   D1   D0  ACK
          ←──────────── 0x75 ───────────────→ ← slave ACK
```

**Why Repeated START (Sr) instead of STOP + START?**

Between Phase 1 (write register address) and Phase 2 (read data), we use a Repeated START instead of STOP + START. This is because:
1. STOP would release the bus, allowing another master to grab it (in multi-master systems)
2. Some slaves reset their internal register pointer on STOP, so we'd lose the address we just set
3. Repeated START is faster (no idle time between)

---

# PART 3 — ADVANCED I2C FEATURES

---

## 3.1 Clock Stretching

The slave can hold SCL LOW to slow down the master. This is "clock stretching."

```
    Normal:
    SCL (by master): ___|‾‾|___|‾‾|___|‾‾|___

    With clock stretching:
    SCL (master releases): ___|‾‾|___|‾‾|____________|‾‾|___
                                                 ↑
                           Slave holds SCL LOW here
                           while it prepares data
                           Master must wait!
```

**Why it exists:** Some slow devices (EEPROMs, ADCs) need time to prepare data after receiving a command. Instead of making the master poll, the slave simply stretches the clock until ready.

**Why it causes problems:**
- The master must monitor SCL after releasing it to check if the slave is stretching
- A buggy slave can hold SCL forever → bus hangs
- Some masters (bit-banged I2C on Raspberry Pi) don't support clock stretching → data corruption
- Linux kernel I2C drivers generally handle it, but you may need to configure timeouts

---

## 3.2 Bus Arbitration — Multi-Master I2C

When two masters try to transmit simultaneously, arbitration decides who wins:

1. Both masters send their START conditions
2. Both put their data bits on SDA simultaneously
3. After each bit, each master reads SDA back
4. If a master sent a 1 (released SDA) but reads back 0 (someone else pulled LOW):
   → That master loses arbitration and backs off
   → The other master doesn't even know there was competition

**This works because of wired-AND**: a 0 on the bus means SOMEONE pulled LOW. If you sent 1 but see 0, another device has higher priority (lower address/data value).

```
    Master A sends: 1  1  0  1  0  ...
    Master B sends: 1  1  0  0  ← B sends 0 while A sends 1
    Bus sees:       1  1  0  0  ← A reads back 0 (it sent 1 → lost!)
    
    Master A detects: "I sent 1 but bus is 0 → I lost arbitration"
    → Master A stops transmitting and tries again later
    → Master B continues unaware of the conflict
```

---

## 3.3 10-Bit Addressing

Standard I2C uses 7-bit addresses (128 possible addresses, but several are reserved → 112 usable). For rare cases needing more than 112 devices on one bus:

```
    10-bit address uses TWO address bytes:
    [1][1][1][1][0][A9][A8][R/W̄]  [A7][A6][A5][A4][A3][A2][A1][A0]
    ←────── first byte ─────────→  ←────── second byte ────────────→
    
    The 11110 prefix is the "10-bit addressing" flag
```

In practice, 10-bit addressing is almost never used. If you're running out of addresses, use an I2C multiplexer (TCA9548A — 8-channel I2C mux with 3 address pins = 8 sub-buses).

---

## 3.4 Bus Stuck Recovery — The 9-Clock Method

**The problem:** A slave device is stuck mid-byte, holding SDA LOW. It's waiting for more clock pulses to finish shifting out its data byte. But the master has stopped clocking (maybe it was reset, or got confused).

Both the master and slave are now stuck:
- Slave holds SDA LOW → master can't generate START (START requires SDA HIGH → LOW)
- Master stopped clocking → slave can't finish its byte and release SDA

```
    Normal:          SDA: ‾‾\_data_/‾ACK‾\_data_/‾‾‾
                     SCL: _|‾|_|‾|_|‾|_|‾|_|‾|_|‾|_

    Stuck mid-byte:  SDA: ‾‾\_data_______________________ (stuck LOW!)
                     SCL: _|‾|_|‾|________________________ (master stopped)
                                     ↑
                              Slave waiting for more clocks
                              to finish its byte
```

**The fix — bit-bang 9 clocks:**

1. Master bit-bangs (toggle GPIO manually) SCL 9 times while leaving SDA released (input mode)
2. The stuck slave receives these clock pulses and shifts out its remaining bits
3. After at most 8 bits, the slave has finished its byte and releases SDA for the ACK
4. On the 9th clock, the slave's ACK phase completes → SDA is released
5. Master now sees SDA HIGH → can generate a proper STOP condition
6. Send STOP → bus is recovered

```c
/* I2C bus recovery — bit-bang on STM32 */
void i2c_bus_recovery(GPIO_TypeDef *scl_port, uint16_t scl_pin,
                      GPIO_TypeDef *sda_port, uint16_t sda_pin)
{
    /* Configure SCL as GPIO output, SDA as GPIO input */
    /* Bit-bang 9 clock pulses */
    for (int i = 0; i < 9; i++) {
        HAL_GPIO_WritePin(scl_port, scl_pin, GPIO_PIN_SET);
        delay_us(5);
        HAL_GPIO_WritePin(scl_port, scl_pin, GPIO_PIN_RESET);
        delay_us(5);
        
        /* Check if SDA is released */
        if (HAL_GPIO_ReadPin(sda_port, sda_pin) == GPIO_PIN_SET) {
            break;  /* Slave released SDA — we're done */
        }
    }
    /* Generate STOP condition */
    HAL_GPIO_WritePin(sda_port, sda_pin, GPIO_PIN_RESET);  /* SDA LOW */
    delay_us(5);
    HAL_GPIO_WritePin(scl_port, scl_pin, GPIO_PIN_SET);    /* SCL HIGH */
    delay_us(5);
    HAL_GPIO_WritePin(sda_port, sda_pin, GPIO_PIN_SET);    /* SDA HIGH while SCL HIGH = STOP */
    
    /* Re-configure pins for I2C peripheral */
}
```

**Zephyr handles this:** The `i2c_recover_bus()` API does exactly this. Some STM32 I2C peripherals also have hardware bus recovery.

---

# PART 4 — I2C SPEED MODES

---

| Mode | Speed | Pull-Up Range | Max Bus Capacitance |
|------|-------|---------------|---------------------|
| Standard | 100 kHz | 4.7kΩ – 10kΩ | 400pF |
| Fast | 400 kHz | 1.5kΩ – 4.7kΩ | 400pF |
| Fast-mode Plus (Fm+) | 1 MHz | 1kΩ – 2.2kΩ | 550pF |
| High-speed (HS) | 3.4 MHz | Special | 400pF |

Most embedded projects use Standard (100kHz) or Fast (400kHz). If you need higher speed, use SPI instead.

---

# PART 5 — COMMON I2C DEVICES

---

| Device | Address | Purpose | Speed |
|--------|---------|---------|-------|
| ICM-42688 / MPU-6050 | 0x68/0x69 | 6-axis IMU | 400kHz |
| TMP102 | 0x48-0x4B | Temperature sensor | 400kHz |
| ADS1115 | 0x48-0x4B | 16-bit ADC | 400kHz |
| 24C02 / AT24C256 | 0x50-0x57 | EEPROM | 400kHz |
| PCA9555 | 0x20-0x27 | 16-bit IO expander | 400kHz |
| TCA9548A | 0x70-0x77 | 8-channel I2C mux | 400kHz |
| SSD1306 | 0x3C/0x3D | OLED display | 400kHz |
| BMP280 | 0x76/0x77 | Pressure/temp sensor | 3.4MHz |

**Address conflicts:** Notice that TMP102 and ADS1115 share the same address range! If you need both, use a TCA9548A mux, or choose devices with different addresses.

---

# PART 6 — I2C vs SMBus

---

SMBus (System Management Bus) is a stricter subset of I2C used in PC motherboards for battery management, fan control, etc.

| Feature | I2C | SMBus |
|---------|-----|-------|
| Bus timeout | None (can hang forever) | 35ms max → auto-release |
| Voltage | Up to 5.5V | 2.5V–5.5V only |
| Clock stretching | Unlimited | Limited to 25ms |
| PEC (error checking) | Optional | Supported (CRC-8) |
| Alert# (interrupt line) | No | Yes |
| Frequency range | 0–400kHz+ | 10kHz–100kHz |

**Why it matters:** If you use an SMBus device on an I2C bus without timeouts, clock stretching bugs can hang the bus forever. Adding a timeout to your I2C driver prevents permanent lockups.

---

# PART 7 — LEVEL SHIFTING FOR I2C

---

I2C level shifting between 3.3V and 5V uses the BSS138 circuit described in chapter 02 (section 3.5). Brief refresher:

```
    3.3V ──┤4.7kΩ├──┬── SDA_3V3    5V ──┤4.7kΩ├──┬── SDA_5V
                     │                              │
                     └──── S ──── G ──── D ─────────┘
                             BSS138
                          (per line)
                          
    Need 2 BSS138 circuits: one for SDA, one for SCL
    The SparkFun BOB-12009 and Adafruit 757 put this on a breakout board
```

**Why I2C level shifting is easy (compared to SPI):** I2C is open-drain, so all drivers only pull LOW or release. The BSS138 circuit works perfectly with this. SPI is push-pull (drives HIGH and LOW), so BSS138 doesn't work — use a dedicated level-shifting IC like TXB0104 for SPI.

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| NACK on every address | Missing pull-ups | Scope SCL/SDA: line never goes HIGH (stays ~0.5V) | Add 4.7kΩ pull-ups to VCC on both SDA and SCL |
| Correct device address, still NACK | Wrong address format (7-bit vs 8-bit notation) | Try address both as-is and shifted right by 1 | Match datasheet format. In Zephyr, use 7-bit address. |
| Bus hangs (SCL or SDA stuck LOW) | Slave stuck mid-byte, or slave crashed | Scope: one line permanently LOW | 9-clock recovery. Power-cycle slave. Check for code bugs. |
| Random data corruption | Wrong I2C speed for pull-up value → slow rise times | Scope rise time on SDA/SCL. Should be < 300ns at 400kHz. | Lower pull-up value (e.g., 2.2kΩ instead of 10kΩ) |
| Works with one device but not two | Address conflict | `i2cdetect` scan — two devices at same address | Use different address pin config, or I2C mux |
| Data OK at 100kHz, corrupt at 400kHz | Rise time too slow for Fast mode | Measure rise time with scope. Calculate RC. | Lower pull-up resistors. Reduce bus capacitance (shorter wires). |
| Slave works but doesn't ACK consistently | Clock stretching not handled by master | Scope: SCL LOW periods vary wildly in duration | Enable clock stretching support in master config |
| 3.3V device damaged after connecting to 5V bus | No level shifter | Dead pin. Measure VCC of all devices. | Add BSS138 level shifter between voltage domains |
| Repeated START doesn't work | Master sends STOP+START instead of Sr | Scope: SDA goes HIGH→LOW between write and read phases | Check HAL/driver settings for repeated start support |
| i2cdetect shows device at wrong address | Address bits set by external pins not as expected | Check A0/A1/A2 pin connections on device | Pull address pins to correct level (GND or VCC) |
| Communication stops after a while | SMBus timeout on slave side | Add logging for NACK/timeout errors. Check slave specs. | Ensure bus transactions complete within 35ms |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── I2C CHEAT SHEET ────────────────────────────────────────────┐
│                                                                                              │
│  CORE:  2 wires (SDA, SCL), open-drain with pull-ups, multi-master, half-duplex            │
│                                                                                              │
│  PULL-UPS (3.3V):                                                                           │
│    100kHz: 4.7kΩ–10kΩ     400kHz: 1.5kΩ–4.7kΩ     1MHz: 1kΩ–2.2kΩ                        │
│    Rise time < 1000ns (Std) / 300ns (Fast) / 120ns (Fm+)                                   │
│    Rise time ≈ 0.8473 × Rp × Cbus                                                          │
│                                                                                              │
│  CONDITIONS:                                                                                 │
│    START:  SDA ↓ while SCL HIGH          STOP: SDA ↑ while SCL HIGH                        │
│    DATA:   SDA stable while SCL HIGH, changes while SCL LOW                                 │
│                                                                                              │
│  ADDRESS BYTE:  [A6..A0][R/W̄]   R/W̄=0 for write, R/W̄=1 for read                          │
│    7-bit addr 0x68 → Write byte = 0xD0 (0x68<<1|0), Read byte = 0xD1 (0x68<<1|1)          │
│                                                                                              │
│  ACK/NACK:  After every 8 data bits (9th clock cycle)                                       │
│    ACK = SDA LOW (receiver)    NACK = SDA HIGH (receiver)                                   │
│                                                                                              │
│  READ REGISTER:  [START][addr+W][ACK][reg][ACK][Sr][addr+R][ACK][data][NACK][STOP]         │
│                                                                                              │
│  BUS RECOVERY:  Bit-bang 9 SCL clocks with SDA released, then STOP                         │
│                                                                                              │
│  LEVEL SHIFT:  BSS138 + pull-ups on both sides (per line). Use for 3.3V↔5V I2C.            │
│                                                                                              │
│  RESERVED ADDRESSES:    0x00 (general call), 0x01-0x07 (reserved),                          │
│    0x78-0x7F (10-bit header, reserved)                                                       │
│                                                                                              │
│  SCAN: i2cdetect -y 1 (Linux), or Zephyr i2c_scan shell command                            │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
