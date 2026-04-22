# 05 — 100Hz SPI Bridge to Jetson
### STM32H743ZI2 (slave) ↔ Jetson (master) · DMA double-buffer · Protobuf frames

**Status:** 🟡 HARDWARE-GATED — fill this in after sessions 02–04  
**Prerequisite:** Sessions 01–04 milestones complete  
**Hardware required:** Nucleo-H743ZI2 · ICM-42688-P breakout · Jetson Orin NX · Saleae Logic 8  
**Unlocks:** `06-ros2-publisher.md`  
**Time budget:** ~6 hours

---

## Goal of This Session

Full integrated 100Hz pipeline:
```
ICM-42688-P ─(I2C 200Hz)→ STM32H743 ─(SPI slave DMA)→ Jetson Orin NX
                                ↑
                     double-buffer with cache maintenance
```

**Milestone**: `rostopic hz /imu/raw` shows **100.0 ± 2.0 Hz**, `ros2 topic echo` shows
physically plausible IMU data, zero EIO errors over a 60-second run.

---

## Frame Format

Design the SPI frame to be:
1. Fixed size (easier for DMA and for Jetson to know how many bytes to clock)
2. Versioned (so old Jetson code doesn't misparse new STM32 frames)
3. Checksum-protected (detect corrupted frames — cache bug will corrupt 1–5% silently)

```c
/* spi_frame.h — shared between STM32 and Jetson */
#pragma once
#include <stdint.h>

#define SPI_FRAME_MAGIC   0xA5
#define SPI_FRAME_VERSION 0x01
#define SPI_FRAME_SIZE    32    /* bytes — fits in one cache line */

typedef struct __attribute__((packed)) {
    uint8_t  magic;              /* 0xA5 — detect frame alignment */
    uint8_t  version;            /* protocol version */
    uint16_t seq;                /* sequence counter — detect dropped frames */
    int16_t  accel_x_raw;        /* ICM-42688 raw ADC counts */
    int16_t  accel_y_raw;
    int16_t  accel_z_raw;
    int16_t  gyro_x_raw;
    int16_t  gyro_y_raw;
    int16_t  gyro_z_raw;
    uint32_t timestamp_us;       /* k_uptime_get_32() in microseconds */
    uint8_t  status;             /* bit 0: IMU valid, bit 1: overrun, bit 2: I2C error */
    uint8_t  _reserved[5];      /* pad to 32 bytes */
    uint16_t crc;                /* CRC-16/CCITT of bytes 0..29 */
} spi_frame_t;

_Static_assert(sizeof(spi_frame_t) == SPI_FRAME_SIZE, "Frame size mismatch");
```

**CRC-16 implementation** (STM32 has hardware CRC, but software is portable):
```c
uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
        }
    }
    return crc;
}
```

---

## STM32 Application Architecture

```
Main thread (100Hz timer)
    ├── Read ICM-42688 over I2C
    ├── Pack into spi_frame_t
    ├── CRC + clean cache
    └── Signal SPI thread: new frame ready

SPI thread (blocked on semaphore)
    ├── Wait for frame-ready semaphore
    ├── Arm SPI DMA with current TX buffer
    ├── Wait for DMA complete (or timeout)
    └── Process RX buffer (Jetson commands)

Double buffers:
    Buffer A: being sent by DMA
    Buffer B: being written by IMU thread
    (swap on every DMA complete)
```

```c
/* main.c — full architecture sketch */
#include <zephyr/kernel.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/i2c.h>
#include "spi_frame.h"
#include "icm42688.h"
#include "crc16.h"

/* Double buffers in DTCM — DMA-accessible, no cache maintenance */
static spi_frame_t __attribute__((section(".dtcm_data"))) tx_buf_a;
static spi_frame_t __attribute__((section(".dtcm_data"))) tx_buf_b;
static spi_frame_t __attribute__((section(".dtcm_data"))) rx_buf;

static volatile spi_frame_t *dma_buf    = &tx_buf_a;  /* DMA owns this */
static volatile spi_frame_t *cpu_buf    = &tx_buf_b;  /* CPU writes this */
static volatile bool          dma_armed = false;

K_SEM_DEFINE(frame_ready, 0, 1);
K_SEM_DEFINE(dma_done,    0, 1);

static uint16_t seq = 0;
static imu_data_raw_t imu_raw;

/* 100Hz timer fires this */
void imu_timer_cb(struct k_timer *timer) {
    /* Read IMU (I2C, non-blocking with k_sem timeout) */
    int ret = icm42688_read_raw(&imu_raw);

    /* Fill CPU buffer */
    cpu_buf->magic        = SPI_FRAME_MAGIC;
    cpu_buf->version      = SPI_FRAME_VERSION;
    cpu_buf->seq          = seq++;
    cpu_buf->accel_x_raw  = imu_raw.ax;
    cpu_buf->accel_y_raw  = imu_raw.ay;
    cpu_buf->accel_z_raw  = imu_raw.az;
    cpu_buf->gyro_x_raw   = imu_raw.gx;
    cpu_buf->gyro_y_raw   = imu_raw.gy;
    cpu_buf->gyro_z_raw   = imu_raw.gz;
    cpu_buf->timestamp_us = k_cyc_to_us_floor32(k_cycle_get_32());
    cpu_buf->status       = (ret == 0) ? 0x01 : 0x04;
    cpu_buf->crc          = crc16_ccitt((uint8_t*)cpu_buf, SPI_FRAME_SIZE - 2);

    k_sem_give(&frame_ready);
}

K_TIMER_DEFINE(imu_timer, imu_timer_cb, NULL);

void spi_thread(void *a, void *b, void *c) {
    struct spi_buf spi_tx = { .buf = (void*)dma_buf, .len = SPI_FRAME_SIZE };
    struct spi_buf spi_rx = { .buf = &rx_buf,         .len = SPI_FRAME_SIZE };
    struct spi_buf_set tx_set = { .buffers = &spi_tx, .count = 1 };
    struct spi_buf_set rx_set = { .buffers = &spi_rx, .count = 1 };

    while (1) {
        /* Wait for new frame from IMU thread */
        k_sem_take(&frame_ready, K_FOREVER);

        /* Swap buffers */
        volatile spi_frame_t *tmp = dma_buf;
        dma_buf = cpu_buf;
        cpu_buf = tmp;
        spi_tx.buf = (void*)dma_buf;

        /* NOTE: using DTCM, no cache flush needed */

        /* Arm DMA and wait for Jetson to clock us */
        int ret = spi_transceive(spi_dev, &spi_cfg, &tx_set, &rx_set);
        if (ret < 0) {
            LOG_ERR("SPI error: %d", ret);
        }
        /* Process rx_buf here if Jetson sends commands */
    }
}

K_THREAD_DEFINE(spi_tid, 2048, spi_thread, NULL, NULL, NULL, 5, 0, 0);

void main(void) {
    verify_imu();
    configure_icm42688();
    k_timer_start(&imu_timer, K_MSEC(10), K_MSEC(10));  /* 100Hz */
    /* threads take over */
}
```

---

## Jetson (Master) Application

```python
#!/usr/bin/env python3
# jetson-spi-master.py — read IMU frames from STM32 at 100Hz
import spidev
import struct
import time
import sys

SPI_FRAME_SIZE = 32
MAGIC = 0xA5
VERSION = 0x01

# struct format: magic(B) version(B) seq(H) ax(h) ay(h) az(h) gx(h) gy(h) gz(h)
#                timestamp(I) status(B) reserved(5x) crc(H)
FRAME_FMT = '<BBHhhhhhhIB5xH'
FRAME_FIELDS = ('magic', 'version', 'seq', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
                'timestamp_us', 'status', 'crc')

ACCEL_SCALE = 9.81 / 16384.0
GYRO_SCALE  = 1.0 / 3753.5

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
        crc &= 0xFFFF
    return crc

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 4_000_000   # 4 MHz
spi.mode = 0

# Dummy TX payload (STM32 ignores this, but SPI is full-duplex)
dummy_tx = [0x00] * SPI_FRAME_SIZE

last_seq = None
errors = 0
total = 0

try:
    while True:
        raw = bytes(spi.xfer2(dummy_tx))
        total += 1

        frame = dict(zip(FRAME_FIELDS, struct.unpack(FRAME_FMT, raw)))

        # Validate
        if frame['magic'] != MAGIC or frame['version'] != VERSION:
            errors += 1
            print(f"[{total}] BAD MAGIC/VERSION: {raw[:4].hex()}", file=sys.stderr)
            continue

        expected_crc = crc16_ccitt(raw[:-2])
        if frame['crc'] != expected_crc:
            errors += 1
            print(f"[{total}] CRC ERROR: got {frame['crc']:04x} expected {expected_crc:04x}",
                  file=sys.stderr)
            continue

        if last_seq is not None and frame['seq'] != (last_seq + 1) & 0xFFFF:
            dropped = (frame['seq'] - last_seq - 1) & 0xFFFF
            print(f"[{total}] DROPPED {dropped} frames (seq {last_seq}→{frame['seq']})",
                  file=sys.stderr)
        last_seq = frame['seq']

        # Scale to physical units
        ax = frame['ax'] * ACCEL_SCALE
        ay = frame['ay'] * ACCEL_SCALE
        az = frame['az'] * ACCEL_SCALE
        gx = frame['gx'] * GYRO_SCALE
        gy = frame['gy'] * GYRO_SCALE
        gz = frame['gz'] * GYRO_SCALE

        print(f"seq={frame['seq']:5d} A:({ax:+.3f},{ay:+.3f},{az:+.3f}) "
              f"G:({gx:+.3f},{gy:+.3f},{gz:+.3f}) errors={errors}/{total}")

        time.sleep(0.01)  # 100Hz

except KeyboardInterrupt:
    print(f"\nFinal error rate: {errors}/{total} ({100*errors/max(total,1):.1f}%)")
    spi.close()
```

---

## Measuring 100Hz Accuracy

```python
# Add to Jetson script: measure actual receive rate
import time

timestamps = []
for _ in range(1000):
    t0 = time.perf_counter()
    spi.xfer2(dummy_tx)
    timestamps.append(time.perf_counter() - t0)
    time.sleep(0.01)

import statistics
print(f"Mean period: {statistics.mean(timestamps)*1000:.3f}ms")
print(f"Stdev:       {statistics.stdev(timestamps)*1000:.3f}ms")
print(f"Max jitter:  {max(timestamps)*1000:.3f}ms")
```

**Target numbers:**
```
Mean period: 10.000ms ± 0.5ms
Stdev:       < 0.3ms
Max jitter:  < 2.0ms  (one-sided)
```

If max jitter exceeds 5ms, the Jetson is not driving SPI transactions at the right rate.
Add a POSIX timer on the Jetson side to fire exactly every 10ms.

---

## Milestone Checklist

- [ ] STM32 logs: "IMU configured, 100Hz timer started"
- [ ] Jetson receives frames with correct MAGIC and VERSION
- [ ] CRC error rate: 0% over 6000 frames (1 minute)
- [ ] Dropped frame rate: 0% over 6000 frames
- [ ] Stationary IMU: accel_z ≈ 9.81 m/s² on Jetson side
- [ ] 100Hz measured on Jetson: mean 10.0ms ± 0.5ms
- [ ] Logic analyzer: no CS glitches, no clock gaps within frame

---

## Pre-Read for Session 6

Before `06-ros2-publisher.md`:
1. Read `zephyr/07_jetson_ros2_bridge.md` fully — publisher node architecture
2. Install `sensor_msgs/Imu` message definition and understand fields
3. Understand `rclpy` QoS profiles: `SENSOR_DATA` vs `RELIABLE`

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Integration Notes
- Time to first valid frame: ___min
- First error encountered: ...

### CRC Test
- Total frames: ___
- CRC errors: ___
- Dropped frames: ___

### Timing
- Mean period: ___ms
- Stdev: ___ms
- Max jitter: ___ms

### Issues Found
- ...

### Changes to Protocol
- (document any changes to spi_frame_t here for posterity)
```
