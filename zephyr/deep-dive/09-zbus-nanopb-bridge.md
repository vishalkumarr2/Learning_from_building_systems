# 09 — ZBus + nanopb: Multi-Sensor Frame Encoding
### STM32H743ZI2 · ZBus message bus · nanopb Protocol Buffers · Variable-length frames

**Status:** 🟡 HARDWARE-GATED  
**Prerequisite:** Sessions `04` (IMU), `07` (CAN), `08` (GPS) milestones complete  
**Hardware required:** All prior hardware (IMU + CAN + GPS active)  
**Unlocks:** `05-100hz-spi-bridge.md` rewrite with nanopb, then `10-jetson-rt-setup.md`  
**Time budget:** ~12 hours  
**Mastery plan:** Project 7

---

## Goal of This Session

Collect IMU + CAN encoder + GPS streams on separate threads. Route all three through ZBus channels. Encode a combined `SensorFrame` protobuf message at 100Hz using nanopb. Verify with a ZBus subscriber that decodes and round-trips the data.

**Milestone**: Terminal shows `encoded=47 bytes` (or similar constant-ish size) at 100Hz. A second ZBus subscriber re-decodes and logs `imu.acc_z≈9.81 wheel_vel=X.X lat=35.67`. Size is identical when all fields are non-zero.

---

## Why nanopb, Not a Raw Struct?

The mastery plan uses nanopb for three reasons:

1. **Forward compatibility**: adding a new field to `.proto` does not break existing Jetson decoders (old bytes decode fine; new fields are ignored).
2. **Language neutrality**: the same `.proto` generates Python stubs for Jetson (via `protoc`), C stubs for STM32 (via `nanopb_generator.py`), and can be consumed by ROS 2 via a bridge.
3. **The zero-field problem forces you to learn framing**: proto3 omits zero-valued fields. This means `pb_encode()` output size varies. Handling this explicitly teaches the most important DMA framing lesson (see Failure Point 2 below).

---

## Step 1: Define the Proto Schema

```proto
// proto/sensor_frame.proto
syntax = "proto3";

message ImuData {
    float acc_x  = 1;
    float acc_y  = 2;
    float acc_z  = 3;
    float gyro_x = 4;
    float gyro_y = 5;
    float gyro_z = 6;
}

message OdomData {
    float  left_vel_mms  = 1;
    float  right_vel_mms = 2;
    uint32 timestamp_ms  = 3;
}

message GpsData {
    double  lat_deg    = 1;
    double  lon_deg    = 2;
    float   alt_m      = 3;
    uint32  satellites = 4;
    bool    has_fix    = 5;
}

message SensorFrame {
    uint32   seq       = 1;   // frame sequence counter
    uint32   ts_ms     = 2;   // STM32 uptime when frame was packed
    ImuData  imu       = 3;
    OdomData odom      = 4;
    GpsData  gps       = 5;
}
```

### nanopb Options File

```
// proto/sensor_frame.options
// Without this, nanopb uses dynamic allocation (not available in Zephyr)
SensorFrame.seq          max_count:1
SensorFrame.ts_ms        max_count:1
ImuData.acc_x            max_count:1
// All fields are scalars — no string/bytes fields, so no max_size needed
// But you MUST specify the max encoded size for the DMA buffer:
SensorFrame              max_size:128
```

### Generate C Stubs

```bash
# From the project root
python3 nanopb_generator.py proto/sensor_frame.proto \
    -I proto/ \
    -D src/generated/

# Produces:
#   src/generated/sensor_frame.pb.h
#   src/generated/sensor_frame.pb.c
```

Add `sensor_frame.pb.c` to `CMakeLists.txt`:
```cmake
target_sources(app PRIVATE
    src/main.c
    src/generated/sensor_frame.pb.c
)
target_include_directories(app PRIVATE src/generated/)
```

---

## Step 2: ZBus Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              STM32H743 ZBus Channels          │
                    │                                               │
   I2C IRQ ──►   imu_thread ──► ZBUS_CHAN: imu_chan (ImuData)      │
   CAN IRQ ──►   can_thread ──► ZBUS_CHAN: odom_chan (OdomData)     │
   UART IRQ ──►  gps_thread ──► ZBUS_CHAN: gps_chan (GpsData)       │
                                                                    │
                    └──────────────────────┬──────────────────────┘
                                           │ subscriber
                              ┌────────────▼────────────┐
                              │        packer_thread      │
                              │  reads all 3 channels     │
                              │  pb_encode() → tx_buf[]   │
                              │  signals DMA              │
                              └───────────────────────────┘
```

### Channel Definitions

```c
/* zbus_channels.h */
#include <zephyr/zbus/zbus.h>
#include "sensor_frame.pb.h"

ZBUS_CHAN_DECLARE(imu_chan);
ZBUS_CHAN_DECLARE(odom_chan);
ZBUS_CHAN_DECLARE(gps_chan);

/* zbus_channels.c */
ZBUS_CHAN_DEFINE(imu_chan,  ImuData,  NULL, NULL, ZBUS_OBSERVERS_EMPTY, ZBUS_MSG_INIT(0));
ZBUS_CHAN_DEFINE(odom_chan, OdomData, NULL, NULL, ZBUS_OBSERVERS_EMPTY, ZBUS_MSG_INIT(0));
ZBUS_CHAN_DEFINE(gps_chan,  GpsData,  NULL, NULL, ZBUS_OBSERVERS_EMPTY, ZBUS_MSG_INIT(0));
```

---

## Step 3: Sensor Threads Publishing to ZBus

```c
/* imu_thread.c — abbreviated, see session 04 for full I2C code */
#include "zbus_channels.h"

K_THREAD_DEFINE(imu_tid, 2048, imu_thread_fn, NULL, NULL, NULL, 5, 0, 0);

void imu_thread_fn(void *p1, void *p2, void *p3) {
    const struct device *i2c = DEVICE_DT_GET(DT_NODELABEL(i2c1));

    while (1) {
        ImuData imu = read_icm42688(i2c);   /* from session 04 */

        zbus_chan_pub(&imu_chan, &imu, K_NO_WAIT);
        /* K_NO_WAIT: if packer is slow, we drop rather than block.
           Sensor reading is more important than waiting for packer. */

        k_sleep(K_USEC(10000));   /* ~100Hz */
    }
}
```

---

## Step 4: Packer Thread

```c
/* packer_thread.c */
#include <zephyr/kernel.h>
#include <zephyr/zbus/zbus.h>
#include <pb_encode.h>
#include "sensor_frame.pb.h"
#include "zbus_channels.h"

LOG_MODULE_REGISTER(packer, LOG_LEVEL_INF);

/* DMA-safe buffer: 32-byte aligned, in DTCM to avoid D-cache issues.
   See session 03 for memory region details. */
static uint8_t __aligned(32) tx_buf[128];

/* Length prefix: 2 bytes before the protobuf payload.
   SPI frame = [len_hi][len_lo][pb_bytes...][padding to 128 bytes]
   Jetson reads the first 2 bytes to know how many subsequent bytes are valid. */
static uint8_t __aligned(32) dma_frame[130];   /* 2-byte header + 128 */

ZBUS_SUBSCRIBER_DEFINE(packer_sub, 4);  /* queue depth 4 */

K_THREAD_DEFINE(packer_tid, 4096, packer_thread_fn, NULL, NULL, NULL, 4, 0, 0);

static uint32_t seq = 0;

void packer_thread_fn(void *p1, void *p2, void *p3) {
    const struct zbus_channel *chan;
    ImuData  imu  = ImuData_init_zero;
    OdomData odom = OdomData_init_zero;
    GpsData  gps  = GpsData_init_zero;

    while (1) {
        /* Wait for any channel to publish */
        if (zbus_sub_wait(&packer_sub, &chan, K_MSEC(15)) != 0) {
            LOG_WRN("Packer timeout — no sensor data for 15ms");
            continue;
        }

        /* Read whichever channel published */
        if (chan == &imu_chan)  zbus_chan_read(&imu_chan,  &imu,  K_NO_WAIT);
        if (chan == &odom_chan) zbus_chan_read(&odom_chan, &odom, K_NO_WAIT);
        if (chan == &gps_chan)  zbus_chan_read(&gps_chan,  &gps,  K_NO_WAIT);

        /* Only encode when the IMU triggers (100Hz pace) */
        if (chan != &imu_chan) continue;

        SensorFrame frame = {
            .seq   = seq++,
            .ts_ms = k_uptime_get_32(),
            .imu   = imu,
            .odom  = odom,
            .gps   = gps,
            .has_imu  = true,
            .has_odom = true,
            .has_gps  = gps.has_fix,  /* only include GPS sub-message if valid */
        };

        /* ── Encode ─────────────────────────────────────── */
        pb_ostream_t stream = pb_ostream_from_buffer(tx_buf, sizeof(tx_buf));
        bool ok = pb_encode(&stream, SensorFrame_fields, &frame);

        if (!ok) {
            LOG_ERR("pb_encode failed: %s", PB_GET_ERROR(&stream));
            continue;
        }

        uint16_t encoded_len = (uint16_t)stream.bytes_written;

        /* ── Length-prefix framing ─────────────────────── */
        /* Store length in first 2 bytes; zero-pad remainder.
           This is the fix for "variable-length proto3 output + fixed DMA size". */
        dma_frame[0] = (encoded_len >> 8) & 0xFF;
        dma_frame[1] =  encoded_len       & 0xFF;
        memcpy(&dma_frame[2], tx_buf, encoded_len);
        memset(&dma_frame[2 + encoded_len], 0, sizeof(dma_frame) - 2 - encoded_len);

        LOG_INF("encoded=%u bytes seq=%u", encoded_len, frame.seq);

        /* Signal SPI DMA thread (from session 05) */
        /* spi_submit_frame(dma_frame, sizeof(dma_frame)); */
    }
}
```

---

## The Three Failure Points (from Mastery Plan, Project 7)

### Failure Point 1: `pb_encode()` returns `false` silently

`pb_encode()` returns `false` and sets an error string via `PB_GET_ERROR(&stream)`. The most common cause is a `string` or `bytes` field missing a `max_size` annotation in the `.options` file.

**Always check the return value:**
```c
if (!pb_encode(&stream, SensorFrame_fields, &frame)) {
    LOG_ERR("pb_encode: %s", PB_GET_ERROR(&stream));
}
```

### Failure Point 2: Variable frame size → stale bytes in DMA buffer

Proto3 omits fields with zero values. When GPS loses fix, `gps.lat_deg=0.0`, `gps.lon_deg=0.0`, etc., are all omitted — the encoded size shrinks. If you DMA-transfer a fixed 128 bytes without tracking the actual encoded length, the Jetson reads `[valid_bytes][stale_bytes_from_previous_frame]`.

**Fix**: length-prefix framing (shown above). The Jetson reads `dma_frame[0]<<8 | dma_frame[1]` to learn how many bytes of `dma_frame[2..]` to pass to the protobuf decoder.

### Failure Point 3: ZBus drops at 100Hz if packer is too slow

If `packer_thread_fn` takes >10ms per iteration, the ZBus subscriber queue fills (depth=4) and oldest notifications are dropped.

**Fix**: raise packer thread priority (`K_THREAD_DEFINE(..., priority=4, ...)`) above sensor threads (priority=5), and enable `CONFIG_ZBUS_MSG_SUBSCRIBER_NET_BUF_DYNAMIC=y` if the queue needs dynamic growth.

---

## Verification: ZBus Decoder Subscriber

Add a second subscriber that re-decodes and confirms round-trip:

```c
ZBUS_SUBSCRIBER_DEFINE(verify_sub, 2);
K_THREAD_DEFINE(verify_tid, 2048, verify_thread_fn, NULL, NULL, NULL, 6, 0, 0);

/* (Define a ZBus channel for encoded frames) */
ZBUS_CHAN_DEFINE(encoded_chan, struct { uint8_t buf[130]; uint16_t len; },
                NULL, NULL, ZBUS_OBSERVERS_EMPTY, ZBUS_MSG_INIT(0));

void verify_thread_fn(void *p1, void *p2, void *p3) {
    const struct zbus_channel *chan;
    while (1) {
        zbus_sub_wait(&verify_sub, &chan, K_FOREVER);
        /* Read encoded frame, decode, log fields */
        /* ... pb_decode() → SensorFrame → LOG_INF acc_z, wheel_vel, lat */
    }
}
```

---

## Milestone Checklist

- [ ] `pb_encode()` returns `true` on every call — no error logs
- [ ] `encoded=XX bytes` printed at 100Hz — stable (varies only when GPS fix state changes)
- [ ] `inject_split` GPS sentence parsed correctly, encoded into SensorFrame
- [ ] Round-trip: verify subscriber logs `imu.acc_z≈9.81`
- [ ] Deliberate zero-field test: disconnect GPS → encoded size shrinks → Jetson still decodes correctly (zero fields, not garbage)
- [ ] ZBus drop counter: 0 over 5-minute run (`CONFIG_ZBUS_CHANNEL_STATISTICS=y`)

---

## Pre-Read for Session 10

Before `10-jetson-rt-setup.md`:
1. `man spidev` and `man 2 ioctl` — you will use `SPI_IOC_MESSAGE`
2. `man cyclictest` — RT latency measurement tool
3. `00-mastery-plan.md` Project 9 failure points: pinmux, speed, spidev_test
4. Linux RT kernel vs PREEMPT_RT patch basics (Jetson Orin uses PREEMPT_RT)

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### nanopb Generation
- nanopb version: ___
- Generated files: sensor_frame.pb.h / sensor_frame.pb.c: yes/no
- `pb_encode()` first success: yes/no

### Encoding
- Frame size (all sensors active, GPS fix): ___ bytes
- Frame size (GPS no fix): ___ bytes
- Difference (zero-field omission observed): yes/no

### ZBus
- Subscriber drop events in 5 min: ___
- Packer thread priority set to: ___

### Issues
- ...
```
