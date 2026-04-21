# SPI — Serial Peripheral Interface

## What Is SPI?

SPI is a **synchronous serial communication protocol** for short-distance communication between a master (controller) and one or more slaves (peripherals).

"Synchronous" means both devices share a **clock signal** — unlike UART which each device generates its own clock and they must agree on speed.

Used for: displays, IMUs, flash memory, SD cards, ADCs, sensor hubs — anything that needs fast, simple wired communication.

---

## Physical Wires — The 4 Lines

```
Master (Jetson / STM32)          Slave (STM32 / sensor)
────────────────────────────────────────────────────────
SCLK  ──────────────────────────►  SCLK   (clock, driven by master)
MOSI  ──────────────────────────►  MOSI   (Master Out Slave In)
MISO  ◄──────────────────────────  MISO   (Master In Slave Out)
CS/NSS ─────────────────────────►  CS     (Chip Select, active LOW)
GND   ────────────────────────────  GND   (always connect ground!)
```

**MOSI**: master sends data to slave on this wire
**MISO**: slave sends data back to master on this wire
**SCLK**: master pulses this to clock bits through
**CS**: master pulls this LOW to select the slave before transfer, HIGH after

---

## How a Transfer Works — Step by Step

```
Master wants to send byte 0b10110101 (0xB5) to slave:

1. Master pulls CS LOW          ← "I'm talking to you"

2. Master drives SCLK LOW (idle state)

3. For each of 8 bits (MSB first):
   a. Master puts bit on MOSI line  (1 or 0)
   b. Slave puts its response bit on MISO
   c. Master pulses SCLK HIGH       ← both sides SAMPLE here
   d. Master pulls SCLK LOW

4. After 8 pulses: 1 byte sent, 1 byte received simultaneously

5. Master pulls CS HIGH         ← "Transfer done"

Timeline:
CS:   ─┐                                              ┌─
       └──────────────────────────────────────────────┘
CLK:       ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐
           └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘
MOSI:  ─1──────0────────1───────1───────0──── ...
MISO:  ─?──────?────────?───────?───────?──── ... (slave responding)
            b7   b6      b5      b4      b3   ...
```

**Key insight**: SPI is **full-duplex** — master and slave send bits simultaneously. Every transfer is an exchange. If you only care about one direction, the other side sends dummy bytes (usually 0x00 or 0xFF).

---

## Multiple Slaves — One Bus, Many CS Lines

```
Master
├── CS0 ──► Slave A (IMU)
├── CS1 ──► Slave B (Flash)
├── CS2 ──► Slave C (Display)
│
MOSI ────────────────────────── (shared)
MISO ────────────────────────── (shared, slaves must tri-state when not selected)
SCLK ────────────────────────── (shared)
```

Only one slave can be active (CS low) at a time. All others float their MISO line (high-impedance).

---

## The 4 SPI Modes — CPOL and CPHA

This is the most confusing part. Each device has a preference for:
- **CPOL** (Clock Polarity): is the clock idle HIGH or LOW?
- **CPHA** (Clock Phase): does data get sampled on the RISING or FALLING edge?

```
Mode 0 (CPOL=0, CPHA=0):  clock idle LOW,  sample on RISING  edge  ← most common
Mode 1 (CPOL=0, CPHA=1):  clock idle LOW,  sample on FALLING edge
Mode 2 (CPOL=1, CPHA=0):  clock idle HIGH, sample on FALLING edge
Mode 3 (CPOL=1, CPHA=1):  clock idle HIGH, sample on RISING  edge
```

Visual for Mode 0 vs Mode 1:
```
Mode 0 (CPHA=0): sample on FIRST edge (rising):
CLK:  ─┐  ┌──      MOSI must be valid BEFORE first edge
       └──┘
         ↑ sample here

Mode 1 (CPHA=1): sample on SECOND edge (falling):
CLK:  ─┐  ┌──      MOSI can change WITH first edge
       └──┘
              ↑ sample here
```

**Rule**: check the sensor datasheet, find its SPI mode, set your controller to match. If mode is wrong, you'll read garbage with no error messages.

Common modes by device:
- ICM-42688 IMU: Mode 0 or Mode 3
- W25Q flash: Mode 0 or Mode 3
- SD cards: Mode 0
- MAX31855 thermocouple: Mode 0

---

## SPI Speed

SPI can run very fast — limited by wire capacitance and buffer drive strength:

| Distance | Max reliable speed |
|---|---|
| On-PCB traces (<5cm) | 50-80 MHz |
| Short cable (<30cm) | 10-20 MHz |
| Long cable (>30cm) | 1-5 MHz |

Our STM32↔Jetson pipeline uses **10 MHz** — appropriate for ~20cm cable with some margin.

At 10 MHz: 1 byte = 800ns. 150-byte frame = 120µs.

---

## SPI vs I2C vs UART — When to Use Which

| | SPI | I2C | UART |
|---|---|---|---|
| Wires | 4 (+ 1 CS per slave) | 2 | 2 |
| Speed | Fast (1-80 MHz) | Slow (100k-3.4 MHz) | Medium (9.6k-921.6k baud) |
| Multi-slave | Yes (one CS per slave) | Yes (7-bit address) | Point-to-point only |
| Full duplex | Yes | No | Yes |
| Distance | Short (<1m) | Very short (<30cm) | Medium (meters with RS-232) |
| Use for | Fast sensors, displays, flash | Slow sensors, config registers | GPS, debug, host PC |

---

## SPI in Zephyr

### Devicetree configuration

```dts
/* app.overlay */
&spi1 {
    status = "okay";
    pinctrl-0 = <&spi1_default>;
    pinctrl-names = "default";

    /* Slave device — IMU on CS pin PA4 */
    imu: icm42688@0 {            /* @0 = CS index 0 */
        compatible = "invensense,icm42688p";
        reg = <0>;               /* CS index */
        spi-max-frequency = <10000000>;  /* 10 MHz */
        /* mode 0 is default (CPOL=0, CPHA=0) */
    };
};
```

### Master transfer (Zephyr API)

```c
#include <zephyr/drivers/spi.h>

static const struct device *spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi1));

static const struct spi_config spi_cfg = {
    .frequency = 10000000,
    .operation = SPI_WORD_SET(8) | SPI_OP_MODE_MASTER | SPI_TRANSFER_MSB,
    /* No .cs here — handled by CS GPIO defined in devicetree */
};

void spi_read_register(uint8_t reg_addr, uint8_t *out, size_t len)
{
    uint8_t tx_buf[1] = { reg_addr | 0x80 };  /* set read bit */
    uint8_t rx_buf[len + 1];

    struct spi_buf tx[] = {{ .buf = tx_buf, .len = 1 }};
    struct spi_buf rx[] = {{ .buf = rx_buf, .len = len + 1 }};
    struct spi_buf_set tx_set = { .buffers = tx, .count = 1 };
    struct spi_buf_set rx_set = { .buffers = rx, .count = 1 };

    spi_transceive(spi_dev, &spi_cfg, &tx_set, &rx_set);

    /* rx_buf[0] is the response to tx_buf[0] — skip it (command echo) */
    memcpy(out, &rx_buf[1], len);
}
```

### Slave mode (Zephyr API)

```c
static const struct spi_config slave_cfg = {
    .operation = SPI_WORD_SET(8) | SPI_OP_MODE_SLAVE,
    .frequency = 0,   /* slave doesn't set frequency — master drives it */
};

void spi_slave_receive(uint8_t *rx_buf, size_t len)
{
    uint8_t tx_dummy[len];    /* send zeros while receiving */
    memset(tx_dummy, 0, len);

    struct spi_buf tx = { .buf = tx_dummy, .len = len };
    struct spi_buf rx = { .buf = rx_buf,   .len = len };
    struct spi_buf_set tx_set = { .buffers = &tx, .count = 1 };
    struct spi_buf_set rx_set = { .buffers = &rx, .count = 1 };

    /* Blocks until master sends `len` bytes */
    spi_transceive(spi_dev, &slave_cfg, &tx_set, &rx_set);
}
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Wrong SPI mode | Read garbage, no errors | Check datasheet CPOL/CPHA, set `spi.mode` to match |
| MSB/LSB wrong | Bits appear reversed | Most devices are MSB-first; check datasheet |
| CS not connected to ground | Random behavior | Always tie CS to GND reference |
| No shared GND | Random bit errors | Connect GND between master and slave |
| Speed too high for cable length | Intermittent errors | Reduce speed or add 33Ω series resistors |
| `xfer` instead of `xfer2` | CS pulses between bytes | Use `xfer2` for multi-byte transfers |
