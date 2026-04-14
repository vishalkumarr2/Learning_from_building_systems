# ZBus — Zephyr Message Bus

## What Problem Does ZBus Solve?

Without ZBus, sharing data between threads requires manual locking:

```c
// WRONG — classic race condition
struct imu_data g_imu;

// Thread A writes
g_imu.accel_x = read_sensor();     // what if Thread B reads mid-write?

// Thread B reads
process(g_imu.accel_x);            // may read half-written struct!
```

With a mutex it's safe, but error-prone and verbose. ZBus wraps this safely with a clean API.

---

## Core ZBus Primitives

| Primitive | What it is | Analogy |
|---|---|---|
| **Channel** | A typed, mutex-protected slot holding the latest value | A mailbox slot |
| **Publisher** | Any thread that calls `zbus_chan_pub()` | Mail sender |
| **Subscriber** | Gets a queue entry when channel changes; reads at own pace | Staff mailbox |
| **Listener** | Callback called synchronously in publisher's thread | Receptionist |

---

## Channel Definition

```c
// sensors/channels.h — declare (usable from any file)
#include <zephyr/zbus/zbus.h>

struct imu_data {
    float accel_x, accel_y, accel_z;
    float gyro_x,  gyro_y,  gyro_z;
    int64_t timestamp_us;
};

struct wheel_data {
    float speed_fl, speed_fr, speed_rl, speed_rr;
    int64_t timestamp_us;
};

ZBUS_CHAN_DECLARE(imu_chan);
ZBUS_CHAN_DECLARE(wheel_chan);
```

```c
// sensors/channels.c — define (exactly once)
ZBUS_CHAN_DEFINE(
    imu_chan,              // C symbol name
    struct imu_data,       // message type — must be plain data (no pointers)
    NULL,                  // validator: fn(chan, msg) returns bool — reject bad data
    NULL,                  // user_data pointer stored on channel
    ZBUS_OBSERVERS_EMPTY,  // static observers (can add dynamically too)
    ZBUS_MSG_INIT(0)       // initial value
);
```

What the macro expands to internally:
- A mutex (or spinlock, configurable with `CONFIG_ZBUS_CHANNEL_ASSUME_LOCKED`)
- A `void *` data pointer to channel's internal memory copy
- Size of the message type
- A linked list of observers

---

## Publish

```c
struct imu_data msg = {
    .accel_x = 1.23f,
    .timestamp_us = k_ticks_to_us_near64(k_uptime_ticks()),
};

// Publish — blocks for up to 1ms to acquire channel mutex
int rc = zbus_chan_pub(&imu_chan, &msg, K_MSEC(1));
if (rc == -EAGAIN) {
    // mutex timeout — channel was held by someone else
}
```

**What happens internally:**
1. Acquires channel mutex (blocks up to timeout)
2. Runs validator (if set) — can reject bad message
3. `memcpy` your struct into channel's internal buffer
4. Notifies all observers (queues entries for subscribers, calls listener callbacks)
5. Releases mutex

**Key**: channel holds only the **latest value**. Publish twice before subscriber reads → first value is overwritten. This is intentional for sensor data — you want fresh, not stale.

---

## Read (peek at latest value without waiting)

```c
struct imu_data data;

// Non-blocking read of latest value
int rc = zbus_chan_read(&imu_chan, &data, K_NO_WAIT);

// Read with timeout (wait for mutex)
rc = zbus_chan_read(&imu_chan, &data, K_MSEC(1));
```

---

## Subscriber — reads at its own pace

Subscriber gets a **message queue entry** when any subscribed channel is updated.

```c
// Define subscriber with queue depth 4
// (if it's 4 slots deep and 5 publishes happen before it reads → oldest dropped)
ZBUS_SUBSCRIBER_DEFINE(imu_sub, 4);

// Register observer on channel (priority 3)
ZBUS_CHAN_ADD_OBS(imu_chan, imu_sub, 3);

// Subscriber thread
void sub_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;
    struct imu_data data;

    while (1) {
        // Block until any subscribed channel has new data
        zbus_sub_wait(&imu_sub, &chan, K_FOREVER);

        // Which channel gave us the notification?
        if (chan == &imu_chan) {
            zbus_chan_read(&imu_chan, &data, K_MSEC(1));
            process_imu(&data);
        }
    }
}

K_THREAD_DEFINE(sub_tid, 1024, sub_thread_fn, NULL, NULL, NULL, 7, 0, 0);
```

**Use subscriber when**: your processing might be slower than publish rate (it queues up), or you want to do work in your own thread.

---

## Listener — synchronous callback

Listener callback runs **inside the publisher's thread context**, synchronously.

```c
static void imu_log_cb(const struct zbus_channel *chan)
{
    struct imu_data data;
    zbus_chan_read(chan, &data, K_NO_WAIT);

    // WARNING: this runs in the publisher's thread!
    // Must be fast — blocking here blocks the publisher
    printk("IMU: ax=%.2f\n", data.accel_x);
}

ZBUS_LISTENER_DEFINE(imu_logger, imu_log_cb);
ZBUS_CHAN_ADD_OBS(imu_chan, imu_logger, 1);
```

**Use listener when**: you need zero-latency reaction AND you can guarantee the callback is fast (microseconds, not milliseconds).

---

## Subscriber vs Listener — Decision Guide

| | Subscriber | Listener |
|---|---|---|
| Runs in | Own thread | Publisher's thread |
| Latency | Higher (queue + thread switch) | Zero (synchronous) |
| Can block | Yes (won't block publisher) | No (blocks publisher!) |
| Can miss data | Yes (queue overflow) | No |
| Use for | Processing, logging, slow operations | Fast reactions (set a flag, write register) |

---

## Multiple Producers to Same Channel

ZBus handles this naturally — mutex prevents concurrent writes:

```c
// Thread A: raw IMU from hardware
zbus_chan_pub(&imu_chan, &raw_imu, K_MSEC(1));

// Thread B: Kalman filter reads raw, publishes filtered
struct imu_data raw, filtered;
zbus_chan_read(&imu_chan, &raw, K_MSEC(1));
kalman_update(&raw, &filtered);
zbus_chan_pub(&imu_filtered_chan, &filtered, K_MSEC(1));
```

---

## Full Example: IMU → Packer

```c
// === imu_thread.c ===
void imu_thread_fn(void *a, void *b, void *c)
{
    const struct device *i2c = DEVICE_DT_GET(DT_NODELABEL(i2c1));
    struct imu_data msg;

    while (1) {
        // Read ICM-42688 over I2C
        uint8_t raw[12];
        i2c_burst_read(i2c, 0x68, 0x1F /* ACCEL_XOUT_H */, raw, 12);

        msg.accel_x = (int16_t)((raw[0] << 8) | raw[1]) / 2048.0f; // ±16g range
        msg.accel_y = (int16_t)((raw[2] << 8) | raw[3]) / 2048.0f;
        msg.accel_z = (int16_t)((raw[4] << 8) | raw[5]) / 2048.0f;
        msg.gyro_x  = (int16_t)((raw[6] << 8) | raw[7]) / 16.4f;   // ±2000dps
        msg.gyro_y  = (int16_t)((raw[8] << 8) | raw[9]) / 16.4f;
        msg.gyro_z  = (int16_t)((raw[10]<< 8) | raw[11])/ 16.4f;
        msg.timestamp_us = k_ticks_to_us_near64(k_uptime_ticks());

        zbus_chan_pub(&imu_chan, &msg, K_NO_WAIT);
        k_msleep(10);  // 100Hz
    }
}

// === packer_thread.c ===
void packer_thread_fn(void *a, void *b, void *c)
{
    struct imu_data imu;
    struct wheel_data wheel;

    while (1) {
        // Grab latest values — non-blocking, use whatever is current
        zbus_chan_read(&imu_chan,   &imu,   K_NO_WAIT);
        zbus_chan_read(&wheel_chan, &wheel, K_NO_WAIT);

        // Encode and swap buffer (see 05_spi_slave_dma.md)
        pack_and_swap(&imu, &wheel);

        k_msleep(10);  // 100Hz
    }
}
```

---

## Enabling ZBus in prj.conf

```kconfig
CONFIG_ZBUS=y
CONFIG_ZBUS_CHANNEL_NAME=y   # optional: name channels for debug
CONFIG_ZBUS_RUNTIME_OBSERVERS=y  # allow ZBUS_CHAN_ADD_OBS at runtime
```

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Listener does slow work (flash write, I2C) | Use subscriber instead |
| `K_NO_WAIT` on pub — channel always busy | Check if something holds mutex too long |
| Reading stale data and assuming it's fresh | Check `timestamp_us` field |
| Pointer in message struct | ZBus copies by value — pointers become dangling |
