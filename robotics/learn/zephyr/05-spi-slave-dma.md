# SPI Slave + DMA + Double Buffering

## Why SPI Slave Is Hard

As a **master** (Jetson) you initiate transfers when you want.
As a **slave** (STM32) the master can start clocking **at any moment**.

```
Jetson (master)          STM32 (slave)
─────────────────────────────────────────────────────
CS  goes LOW  ──────►  [DMA must already have TX buffer loaded!]
CLK starts    ──────►  Hardware shifts bits out of SPI_DR register
Data flows    ◄──────  If DMA not ready → sends 0xFF or stale data
CS  goes HIGH ──────►  Transfer done, Jetson has whatever was in SPI_DR
```

**The core constraint**: from the moment CS goes low to the first CLK edge, you may have only **~100ns to 1µs** depending on Jetson configuration. The CPU cannot react that fast to load a buffer — **DMA must already be armed**.

---

## Double Buffering — The Solution

```
Two buffers in memory:

Buffer A (DMA active):  ████████████████  ← DMA is sending this RIGHT NOW
Buffer B (CPU idle):    ░░░░░░░░░░░░░░░░  ← CPU/packer is writing next frame here

10ms tick:
  t=1ms:  packer finishes writing Buffer B
  t=1ms:  SWAP: B becomes active, A becomes idle (atomic pointer swap)
  t=5ms:  Jetson asserts CS
  t=5ms:  CS ISR: DMA already points to Buffer B → starts sending
  t=5.05ms: Transfer complete
  t=5ms-10ms: packer writes next frame into Buffer A (now idle)
  t=10ms: SWAP again
```

No race condition — packer always writes to the buffer DMA is NOT reading.

---

## Complete Implementation

### Buffer definitions

```c
// comms/spi_slave.c
#include <zephyr/kernel.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include "crc16.h"

#define FRAME_MAX_BYTES  300

// DMA-accessible region — annotate for linker to place in correct SRAM bank
// On STM32H7: DMA1/2 can only access D2 SRAM, not D1 SRAM — check your chip!
static uint8_t __aligned(32) tx_buf[2][FRAME_MAX_BYTES];
static volatile int     tx_active = 0;   // which buffer DMA is currently sending
static volatile uint16_t tx_len   = 0;   // valid bytes in active buffer

static K_SEM_DEFINE(frame_ready, 0, 1);  // signals SPI task new frame available
```

### Frame packing — called by packer thread

```c
// Builds a framed packet in the IDLE buffer, then atomically swaps
// Frame format: [0xAA][len_hi][len_lo][...payload...][crc_hi][crc_lo]
void spi_slave_push_frame(const uint8_t *payload, uint16_t len)
{
    if (len + 5 > FRAME_MAX_BYTES) {
        return;  // payload too large
    }

    int write = 1 - tx_active;   // write to the buffer DMA is NOT using

    // Build frame in idle buffer
    tx_buf[write][0] = 0xAA;
    tx_buf[write][1] = (len >> 8) & 0xFF;
    tx_buf[write][2] = len & 0xFF;
    memcpy(&tx_buf[write][3], payload, len);

    // CRC over payload only (not header)
    uint16_t crc = crc16_ccitt(&tx_buf[write][3], len);
    tx_buf[write][3 + len]     = (crc >> 8) & 0xFF;
    tx_buf[write][3 + len + 1] = crc & 0xFF;

    // Cache flush: ensure CPU cache is written to SRAM before DMA reads
    // Required on Cortex-M7 (STM32H7/F7) — omit for M0/M3/M4 (no D-cache)
    SCB_CleanDCache_by_Addr((uint32_t*)tx_buf[write], len + 5);

    // Atomic swap — next CS edge will send this buffer
    tx_len   = len + 5;
    tx_active = write;           // single word write = atomic on ARM

    k_sem_give(&frame_ready);
}
```

### CS falling-edge interrupt — reload DMA

```c
static const struct gpio_dt_spec cs_gpio =
    GPIO_DT_SPEC_GET(DT_NODELABEL(spi1_cs), gpios);

static struct gpio_callback cs_cb_data;

// Called when Jetson asserts CS low
static void cs_falling_isr(const struct device *dev,
                            struct gpio_callback *cb, uint32_t pins)
{
    // Reconfigure SPI DMA TX to point at current active buffer
    // This is the time-critical path — must complete << 1µs
    spi_dma_reload(tx_buf[tx_active], tx_len);
}

void spi_slave_init(void)
{
    gpio_pin_configure_dt(&cs_gpio, GPIO_INPUT);
    gpio_init_callback(&cs_cb_data, cs_falling_isr, BIT(cs_gpio.pin));
    gpio_add_callback(cs_gpio.port, &cs_cb_data);
    gpio_pin_interrupt_configure_dt(&cs_gpio, GPIO_INT_EDGE_FALLING);
}
```

### Devicetree overlay (app.overlay)

```dts
&spi1 {
    status = "okay";
    pinctrl-0 = <&spi1_slave_default>;

    /* SPI slave mode — no cs-gpios here (master controls CS) */
    /* DMA channels assigned in board's DTS or here */
    dmas = <&dma1 3 3 0x10>, <&dma1 2 3 0x10>;
    dma-names = "tx", "rx";
};

/ {
    /* Expose CS pin as GPIO so we can interrupt on it */
    spi1_cs: spi1-cs {
        compatible = "gpio-keys";
        gpios = <&gpioa 4 GPIO_ACTIVE_LOW>;
    };
};
```

---

## SPI Framing Protocol

```
Offset  Size    Content
──────────────────────────────────────────────
0       1       Sync byte: 0xAA
1-2     2       Payload length (big-endian uint16)
3..N    N       nanopb-encoded SensorFrame
N+1..N+2  2     CRC16-CCITT over payload bytes only
```

**Why the sync byte?**
If Jetson misses a byte (glitch, reboot), the byte positions shift. The sync byte lets Jetson re-lock:

```python
# Jetson sync recovery — keep reading until 0xAA seen
while True:
    b = spi.readbytes(1)[0]
    if b == 0xAA:
        break
# read length, then payload
```

**Why CRC?**
SPI has no built-in error detection. Electrical noise on a long cable, crosstalk with other signals, or a voltage glitch can flip bits silently. CRC catches this and lets you discard the frame rather than process garbage sensor data.

---

## DMA Register Details (STM32 HAL style, for reference)

```c
// What spi_dma_reload() does internally:
void spi_dma_reload(const uint8_t *buf, uint16_t len)
{
    // Disable DMA stream
    DMA1_Stream3->CR &= ~DMA_SxCR_EN;
    while (DMA1_Stream3->CR & DMA_SxCR_EN) {}  // wait for disable

    // Set new source address and count
    DMA1_Stream3->M0AR = (uint32_t)buf;
    DMA1_Stream3->NDTR = len;

    // Re-enable
    DMA1_Stream3->CR |= DMA_SxCR_EN;

    // Enable SPI
    SPI1->CR1 |= SPI_CR1_SPE;
}
```

On a higher-level RTOS like Zephyr this is hidden inside `spi_transceive()`, but the principle is the same.

---

## Timing Constraints Summary

| Event | Time budget | What happens if violated |
|---|---|---|
| CS ISR → DMA armed | < 1µs | First bytes sent as 0xFF (corrupt frame) |
| Packer → buffer swap | < 9ms (before next CS) | Old frame sent again (stale, but valid) |
| Cache flush before swap | Before DMA starts | DMA reads old cached data (silent corruption) |
| CRC generation | Part of packer time | Skip it and you can't detect noise corruption |

---

## Debugging Tips

- **Scope CS + CLK + MOSI** — verify timing matches your spi_slave config (CPOL/CPHA)
- **Scope CS + DMA TC interrupt** — DMA completion should happen before CS goes high
- **Add sequence number in frame** (`SensorFrame.seq`) — Jetson can detect dropped frames
- **Log `is_corrupt`** on Jetson — how often does CRC fail? >0.1% = check wiring
- **Cache bug symptom**: first ~32 bytes correct, rest is old/garbage → flush cache
