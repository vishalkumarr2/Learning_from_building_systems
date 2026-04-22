# 03 — DMA + D-Cache Gotchas
### STM32H743ZI2 · Cortex-M7 D-cache · SPI DMA at 400Hz

**Status:** 🟡 HARDWARE-GATED — fill this in after `02` is working  
**Prerequisite:** `02-spi-slave-first-frame.md` milestone checklist complete  
**Hardware required:** Same as `02` + Saleae Logic 8  
**Unlocks:** `04-imu-i2c-reads.md`  
**Time budget:** ~5 hours

---

## Goal of This Session

Deliberately reproduce the D-cache coherency bug, confirm it on the logic analyzer, then
fix it. You'll never forget the fix once you've seen it bite in hardware.

**Milestone**: SPI slave running at 100Hz with DMA, zero corrupted frames over a 10-second run.

---

## Why This Session Exists

The STM32H743 has a **32KB D-cache** enabled by default in the Zephyr Nucleo H743 board config.
DMA bypasses the cache and writes directly to SRAM. The CPU (and your code) reads from cache,
which may still hold stale data from before the DMA write.

This is invisible in the basic `spi_transceive` approach because `spi_transceive` is blocking
and synchronous — Zephyr's SPI driver handles the cache flushing internally.

Once you add **DMA double-buffering** for 100Hz operation, *you* are responsible for the cache.

---

## Theory: D-Cache Coherency in 5 Minutes

```
Memory layout:
┌───────────────────────────────────────────────┐
│  DTCM (0x2000_0000)   ← Tightly Coupled, no cache, DMA-accessible  │
│  AXI SRAM (0x2400_0000) ← Behind D-cache, DMA-accessible          │
│  SRAM1 (0x3000_0000)  ← Behind D-cache, DMA-accessible            │
└───────────────────────────────────────────────┘

D-cache sits between CPU and all SRAM except DTCM.
DMA controller has its own bus master — bypasses cache entirely.

WRITE path (CPU → DMA):
  CPU writes tx_buf[0] = 0xAA → goes into cache (dirty line)
  DMA starts SPI TX — reads tx_buf[0] from SRAM directly (0x00 if cache not flushed)
  Result: Jetson receives 0x00 instead of 0xAA

READ path (DMA → CPU):
  DMA writes received byte into rx_buf[0] = 0x11 → SRAM directly
  CPU reads rx_buf[0] → gets 0x?? from cache (whatever was there before)
  Result: CPU sees stale data, DMA wrote the correct value but cache hides it
```

**Fix: cache maintenance before DMA**
```c
// Before DMA TX (flush CPU writes to SRAM):
SCB_CleanDCache_by_Addr((uint32_t*)tx_buf, sizeof(tx_buf));

// After DMA RX completes (invalidate stale cache):
SCB_InvalidateDCache_by_Addr((uint32_t*)rx_buf, sizeof(rx_buf));
```

**Alternative: put buffers in DTCM**
```c
// Place buffers in DTCM (no cache, no maintenance needed):
static uint8_t tx_buf[64] __attribute__((section(".dtcm_data")));
static uint8_t rx_buf[64] __attribute__((section(".dtcm_data")));
```

DTCM is zero-latency for the CPU, 32-bit wide, but limited to 128KB. Use it for
time-critical DMA buffers; leave SRAM for larger allocations.

---

## Experiment 1: Reproduce the Bug

### Setup

Build the double-buffered SPI slave from session `02` but **without** cache maintenance:

```c
/* INTENTIONALLY BUGGY — no cache maintenance */
static uint8_t tx_buf_a[8] = { 0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04 };
static uint8_t tx_buf_b[8] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };

void spi_transfer_complete_cb(const struct device *dev, int result, void *user_data) {
    /* Swap buffers */
    current_buf = (current_buf == tx_buf_a) ? tx_buf_b : tx_buf_a;
    /* ARM NEXT TRANSFER — no cache flush */
    arm_next_spi_transfer(current_buf);
}
```

### What to Observe

Run the Pi master sending 100 transfers:
```python
errors = 0
for i in range(100):
    result = spi.xfer2([i & 0xFF] * 8)
    expected = KNOWN_PATTERN
    if result != expected:
        errors += 1
        print(f"Frame {i}: expected {expected}, got {result}")
print(f"Error rate: {errors}/100")
```

**Expected with bug**: 0–5% error rate, intermittent, non-deterministic.
Logic analyzer will show corrupted frames intermittently.

**Key insight**: the bug is intermittent because D-cache uses write-back eviction.
Sometimes the cache line happens to be evicted before DMA reads it (correct).
Sometimes it isn't (corrupted). Deterministic in temperature and load, not in general.

---

## Experiment 2: Fix the Bug

### Fix A: Cache Maintenance

```c
void arm_next_spi_transfer(uint8_t *buf, size_t len) {
    /* Flush CPU's dirty cache lines to SRAM before DMA reads them */
    SCB_CleanDCache_by_Addr((uint32_t *)buf, len);

    /* Re-arm DMA */
    struct spi_buf spi_buf = { .buf = buf, .len = len };
    struct spi_buf_set buf_set = { .buffers = &spi_buf, .count = 1 };
    spi_transceive_cb(spi_dev, &spi_cfg, &buf_set, NULL, spi_transfer_complete_cb, NULL);
}

void process_received(uint8_t *buf, size_t len) {
    /* Invalidate cache before CPU reads DMA-written data */
    SCB_InvalidateDCache_by_Addr((uint32_t *)buf, len);

    /* Now safe to read */
    LOG_INF("RX[0] = 0x%02x", buf[0]);
}
```

**Note on alignment**: `SCB_CleanDCache_by_Addr` requires address aligned to 32 bytes
(D-cache line size). Use `__aligned(32)` attribute on buffers.

```c
/* Aligned buffer declaration */
static uint8_t __aligned(32) tx_buf_a[64];
static uint8_t __aligned(32) tx_buf_b[64];
```

### Fix B: DTCM Placement (Better for Hot Paths)

```c
/* Linker script overlay — add to boards/nucleo_h743zi2.overlay */
/*
   or use attribute directly:
*/
static uint8_t tx_buf_a[64] __attribute__((section(".dtcm_data")));
static uint8_t tx_buf_b[64] __attribute__((section(".dtcm_data")));

/* No cache maintenance needed — DTCM bypasses D-cache */
```

Check in `zephyr/boards/arm/nucleo_h743zi2/nucleo_h743zi2.ld` that `.dtcm_data` section
is defined pointing to `0x2000_0000`.

---

## Experiment 3: Measure the 100Hz Timing Budget

At 100Hz, each SPI transaction must complete in under 10ms. Measure actual timing:

```c
/* Add GPIO toggle around SPI transfer for scope/LA measurement */
#include <zephyr/drivers/gpio.h>

const struct gpio_dt_spec timing_pin = GPIO_DT_SPEC_GET(DT_ALIAS(timing), gpios);

void arm_next_spi_transfer(uint8_t *buf, size_t len) {
    gpio_pin_set_dt(&timing_pin, 1);   /* GPIO HIGH = SPI transaction active */
    SCB_CleanDCache_by_Addr((uint32_t *)buf, len);
    /* ... arm DMA ... */
}

void spi_transfer_complete_cb(...) {
    gpio_pin_set_dt(&timing_pin, 0);   /* GPIO LOW = transaction done */
    SCB_InvalidateDCache_by_Addr((uint32_t *)rx_buf, sizeof(rx_buf));
    /* ... process, prepare next buf ... */
}
```

Connect timing pin (e.g., PB0) to Logic 8 Ch4. Measure:
- Transfer duration (HIGH pulse width)
- Gap between transfers
- Jitter over 100 consecutive transfers

**Target numbers:**
```
8-byte frame @ 4MHz:  duration = 8 × 8 / 4e6 = 16µs
Cache flush:          ~1µs per 32-byte cache line
DMA setup overhead:   ~5µs
Callback overhead:    ~2µs
Total per frame:      < 50µs (budget is 10ms at 100Hz — plenty of headroom)
```

---

## Milestone Checklist

- [ ] Bug reproduced: intermittent corrupted frames visible on logic analyzer
- [ ] Fix A applied: error rate drops to 0/100
- [ ] Fix B applied: DTCM buffers eliminate cache maintenance calls
- [ ] Timing measured: GPIO toggle visible on Ch4, duration < 50µs per frame
- [ ] 10-second run at 100Hz: 0 frame errors, no HardFaults
- [ ] `kernel stacks` in Zephyr shell: no stack overflows

---

## Pre-Read for Session 4

Before `04-imu-i2c-reads.md`:
1. Read `zephyr/09_i2c.md` — specifically the `i2c_write_read` pattern and stuck bus recovery
2. ICM-42688-P datasheet: Section 14 (SPI interface), Section 15 (I2C interface), Register Map Appendix
3. `00-mastery-plan.md` "Stuck I2C Bus After Power Cycle"

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Bug Reproduction
- Error rate before fix: ___/100 frames
- LA screenshot filename: ...
- What the corrupted bytes looked like: ...

### Fix Results
- Fix A (cache maintenance): error rate → ___/100
- Fix B (DTCM): error rate → ___/100

### Timing Measurements
- Transfer duration: ___µs
- Jitter: ±___µs
- Overhead budget remaining: ___µs / 10000µs

### Surprises
- ...

### Questions for Session 4
- ...
```
