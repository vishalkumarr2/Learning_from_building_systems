# 02 — SPI Slave: First Frame Capture
### STM32H743ZI2 as SPI slave · Saleae Logic 8 · Jetson (or Raspberry Pi) as master

**Status:** 🟡 HARDWARE-GATED — fill this in when logic analyzer arrives  
**Prerequisite:** `01-first-build-flash-debug.md` completed + working GDB  
**Hardware required:** Nucleo-H743ZI2 · Saleae Logic 8 · a SPI master (Raspberry Pi or Jetson)  
**Unlocks:** `03-dma-cache-gotchas.md`  
**Time budget:** ~4 hours

---

## Goal of This Session

Walk away having:
1. STM32 running as SPI slave, sending a known payload
2. Saleae Logic 8 capturing that payload with SPI decode
3. Raspberry Pi / Jetson reading those bytes via `spidev`
4. All 3 sides agreeing on the same bytes

That last point is the milestone. Once you have "STM32 sends X → logic analyzer sees X → Jetson reads X," everything else is just adding more bytes and faster clocks.

---

## Theory Review (Read Before Wiring)

### SPI Roles and Timing

```
              MOSI ─────────────────────────────────►
Master ──── CS   ─────┐     ┌──────────────────── Slave (STM32)
(Jetson)    CLK ──────┼─────┼────────────────────
              MISO ◄──┴─────┴────────────────────
```

**Slave constraint**: the slave must have TX buffer ready *before* CS goes low.
The clock starts immediately after CS asserts — no time to prepare.

See: `zephyr/05_spi_slave_dma.md` — "SPI Slave Pre-Arming Race" — this is the core bug
you'll hit if you try to arm the DMA in the CS interrupt callback.

### CPOL/CPHA Modes

| Mode | CPOL | CPHA | Clock idle | Sample edge |
|------|------|------|-----------|-------------|
| 0 | 0 | 0 | Low | Rising |
| 1 | 0 | 1 | Low | Falling |
| 2 | 1 | 0 | High | Falling |
| 3 | 1 | 1 | High | Rising |

**Use Mode 0 (CPOL=0, CPHA=0) for first bringup.** It's the most common and easiest to
verify on a logic analyzer — the clock idles low, making it visually obvious if a wire is disconnected.

---

## Wiring Diagram

```
STM32 Nucleo-H743ZI2          Raspberry Pi 4 (or Jetson)
─────────────────────────     ──────────────────────────
SPI1_SCK  = PA5 (CN7 pin 10)  GPIO11 (SPI0_CLK) = pin 23
SPI1_MISO = PA6 (CN7 pin 12)  GPIO9  (SPI0_MISO) = pin 21
SPI1_MOSI = PA7 (CN7 pin 14)  GPIO10 (SPI0_MOSI) = pin 19
SPI1_NSS  = PA4 (CN7 pin 17)  GPIO8  (SPI0_CE0)  = pin 24

GND (CN6 pin 6)               GND (pin 6)

⚠ Keep wires under 15cm. Run GND alongside CLK.
⚠ Both boards powered from the same USB hub OR add GND wire between them.
⚠ Check voltage: Jetson GPIO = 3.3V. Nucleo GPIO = 3.3V. ✓
```

**Saleae Logic 8 connections:**
```
Logic 8 Ch0  → SPI1_SCK  (PA5)
Logic 8 Ch1  → SPI1_CS   (PA4)
Logic 8 Ch2  → SPI1_MISO (PA6)
Logic 8 Ch3  → SPI1_MOSI (PA7)
Logic 8 GND  → STM32 GND (CN6 pin 6) — REQUIRED, not optional
```

---

## Step-by-Step: Minimum Working SPI Slave

### Step 1: STM32 SPI Slave Application

```c
/* zephyr/deep-dive/firmware/01-spi-slave-minimal/src/main.c */
#include <zephyr/kernel.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(spi_slave, LOG_LEVEL_DBG);

/* SPI device from devicetree */
static const struct device *spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi1));

static const struct spi_config spi_cfg = {
    .frequency = 1000000,                /* 1 MHz — slow for bringup */
    .operation = SPI_WORD_SET(8) |       /* 8-bit words */
                 SPI_TRANSFER_MSB |      /* MSB first */
                 SPI_OP_MODE_SLAVE,      /* <-- slave mode */
    .slave = 0,
    .cs = NULL,                          /* hardware CS managed by SPI peripheral */
};

/* Known payload — easy to verify visually */
static uint8_t tx_buf[] = { 0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04 };
static uint8_t rx_buf[8];

void main(void) {
    if (!device_is_ready(spi_dev)) {
        LOG_ERR("SPI device not ready");
        return;
    }
    LOG_INF("SPI slave ready");

    struct spi_buf tx_spi_buf = { .buf = tx_buf, .len = sizeof(tx_buf) };
    struct spi_buf rx_spi_buf = { .buf = rx_buf, .len = sizeof(rx_buf) };
    struct spi_buf_set tx_set = { .buffers = &tx_spi_buf, .count = 1 };
    struct spi_buf_set rx_set = { .buffers = &rx_spi_buf, .count = 1 };

    while (1) {
        LOG_INF("Waiting for SPI master...");
        int ret = spi_transceive(spi_dev, &spi_cfg, &tx_set, &rx_set);
        if (ret < 0) {
            LOG_ERR("spi_transceive error: %d", ret);
        } else {
            LOG_INF("TX sent; RX received: %02x %02x %02x %02x",
                    rx_buf[0], rx_buf[1], rx_buf[2], rx_buf[3]);
        }
    }
}
```

**Devicetree overlay** (`boards/nucleo_h743zi2.overlay`):
```dts
&spi1 {
    status = "okay";
    pinctrl-0 = <&spi1_sck_pa5 &spi1_miso_pa6 &spi1_mosi_pa7 &spi1_nss_pa4>;
    pinctrl-names = "default";
};
```

### Step 2: Raspberry Pi Master (spidev)

```python
#!/usr/bin/env python3
# pi-spi-master.py — Send known bytes, print what slave returned
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)             # bus 0, device 0 (CE0)
spi.max_speed_hz = 1000000  # 1 MHz — match STM32 config
spi.mode = 0                # CPOL=0, CPHA=0

MASTER_TX = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]

while True:
    result = spi.xfer2(MASTER_TX)
    print(f"Sent:     {[hex(b) for b in MASTER_TX]}")
    print(f"Received: {[hex(b) for b in result]}")
    print(f"Expected: ['0xaa', '0xbb', '0xcc', '0xdd', '0x1', '0x2', '0x3', '0x4']")
    time.sleep(1)
```

**Verify:**
- Pi prints `0xaa 0xbb 0xcc 0xdd 0x01 0x02 0x03 0x04` ← matches `tx_buf` in STM32 code
- Logic analyzer shows 8 bytes with those values on MISO channel

### Step 3: Logic Analyzer Setup

In **Saleae Logic 2**:
1. Set sample rate: **50 MS/s** (50× the 1MHz SPI clock — minimum is 10×)
2. Add **SPI Analyzer**: Channels → Ch0=CLK, Ch1=CS, Ch2=MISO, Ch3=MOSI · CPOL=0 · CPHA=0
3. **Trigger**: Ch1 (CS) falling edge
4. Click **Start** — Logic 2 arms the trigger
5. Run the Pi script — trigger fires on CS assert
6. Stop capture — you should see 8 decoded bytes

**What to look for:**
```
Decoded SPI transaction:
CS  ─────┐                              ┌───
         └──────────────────────────────┘
CLK ─────┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌
MISO  0xAA    0xBB    0xCC    0xDD    0x01 ...
MOSI  0x11    0x22    0x33    0x44    0x55 ...
```

---

## The First Bug You Will Hit

**Symptom**: Pi receives `0x00 0x00 0x00 0x00 ...` instead of `0xAA 0xBB ...`

**Cause**: `spi_transceive` in Zephyr slave mode blocks until the master initiates. But you
called it *after* starting the loop print, so it isn't armed when CS drops. The hardware
sends the default 0x00 from the empty shift register.

**Fix**: Arm `spi_transceive` BEFORE signaling ready. In production, use double-buffering
and arm on the previous transfer-complete callback. For this exercise, just ensure the
`spi_transceive` call is reached before the Pi runs `spi.xfer2`.

Add a 200ms delay in the Pi script before the first transfer:
```python
time.sleep(0.2)   # give STM32 time to arm before first CS
result = spi.xfer2(MASTER_TX)
```

---

## Milestone Checklist

- [ ] Logic analyzer captures SPI clock at correct frequency
- [ ] Decoded MISO shows `AA BB CC DD 01 02 03 04`
- [ ] Decoded MOSI shows `11 22 33 44 55 66 77 88`
- [ ] Pi Python script prints matching received bytes
- [ ] STM32 LOG shows `RX received: 11 22 33 44`
- [ ] Logic analyzer shows CS held low for entire 8-byte transaction
- [ ] Speed bumped to 4 MHz and all bytes still correct (signal integrity check)

---

## Pre-Read for Session 3

Before `03-dma-cache-gotchas.md`:
1. Re-read `zephyr/06_dma.md` — DMA scatter-gather and double-buffering
2. Re-read `00-mastery-plan.md` "D-Cache Coherency (2 days lost if unknown)"
3. Look up STM32H743 reference manual Section 14 (L1 cache): `SCB_CleanDCache_by_Addr`, `SCB_InvalidateDCache_by_Addr`

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Wiring Notes
- Wire lengths used: CLK=___cm, CS=___cm, MISO=___cm, MOSI=___cm
- Common GND: yes/no — problem if no

### First Capture
- Time to first valid capture: ___min
- First bug encountered: ...
- Fix: ...

### Logic Analyzer Observations
- Actual SPI clock frequency measured: ___Hz (should be ~1MHz)
- Any ringing visible on CLK? (screenshot if yes)
- CS glitches? (screenshot if yes)

### Speed Sweep Results
| Speed | Correct? | Notes |
|-------|----------|-------|
| 1 MHz | | |
| 4 MHz | | |
| 8 MHz | | |

### Questions for Next Session
- ...
```
