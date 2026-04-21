# Exercises: ZBus + nanopb

---

## Section A — Conceptual Questions

**Q1.** You have one thread reading from an I2C accelerometer and another thread building SPI frames. Without ZBus, what specific failure can occur when both threads access the same global `struct imu_data`? Describe a concrete scenario where `accel_x`, `accel_y`, and `accel_z` are internally inconsistent in a single read.

<details><summary>Answer</summary>

ARM Cortex-M has no atomic multi-word copy. Reading/writing a struct is a series of separate 32-bit loads/stores. If the ARM scheduler preempts the writer mid-write — for example, after writing `accel_x` and `accel_y` but before writing `accel_z` — and the reader runs at that moment, the reader gets `accel_x` and `accel_y` from the current measurement, but `accel_z` from the **previous** measurement. The struct is internally inconsistent: it describes a physical state that never existed. The data looks plausible (no garbage values, just a mix of two timestamps), making it very hard to detect.

</details>

---

**Q2.** Explain the difference between a ZBus **subscriber** and a ZBus **listener** in terms of:
- Which thread their code runs in
- Whether they can call `k_msleep()`
- Whether they can miss a message

<details><summary>Answer</summary>

**Subscriber:**
- Runs in its **own dedicated thread** (defined with `K_THREAD_DEFINE`).
- CAN call `k_msleep()`, wait on semaphores, do I2C writes, etc. — it is a normal thread.
- CAN miss messages if the subscriber queue fills up (e.g. queue depth 4, but 5 messages come in before the subscriber reads). The oldest notification is dropped.

**Listener:**
- Runs **inside the publisher's thread context**, called synchronously during `zbus_chan_pub()`.
- MUST NOT call `k_msleep()` or any blocking function — doing so blocks the publisher for the entire sleep duration.
- CANNOT miss messages — the callback fires on every publish, before `pub()` returns.

</details>

---

**Q3.** What does proto3 do with a field that equals its default value during encoding? Give the concrete example of what happens to a `SensorFrame` message when the GPS loses fix and all GPS position fields become zero.

<details><summary>Answer</summary>

Proto3 **omits** any field whose value equals the type's default. For numeric types the default is `0`. For booleans it is `false`. For strings it is `""`.

When GPS loses fix, `latitude=0.0`, `longitude=0.0`, `altitude_m=0.0f`, `satellites=0` are all zero — all omitted. Even `has_fix=false` (default for bool) is omitted. The entire `gps` sub-message may encode to 0 bytes.

This means `pb_encode()` produces a **shorter output** than when GPS had a fix. If the SPI DMA frame is sized for the maximum (128 bytes) and you always DMA-transfer 128 bytes, the Jetson reads the valid protobuf bytes PLUS trailing bytes from the **previous DMA transfer** (stale data still in the DMA buffer). The decoder interprets those stale bytes as additional fields → silent data corruption.

</details>

---

**Q4.** The ZBus channel holds only the **latest value**. Explain what this means for a subscriber publishing at 100Hz. If the packer thread runs at only 50Hz (once every 20ms), does it receive every message published in those 20ms? What does it receive?

<details><summary>Answer</summary>

No. The packer subscriber receives **only the most recent value** published when it finally reads the channel. In 20ms at 100Hz, two messages were published — the packer reads only the second one. The first is silently overwritten.

The subscriber's **queue** only stores the *notification* (a pointer to the channel), not the data itself. So even if the queue depth is 4, the actual data for intermediate publishes is gone. When the subscriber wakes up and calls `zbus_chan_read()`, it always gets the *current* value in the channel at that moment.

For sensor data this is typically fine (you want fresh, not stale), but for commands or events where every message matters, you must size the queue carefully and use a different pattern (e.g. a k_msgq with the full message).

</details>

---

**Q5.** What does `ZBUS_OBSERVER_DEFINE` (and its siblings `ZBUS_SUBSCRIBER_DEFINE`, `ZBUS_LISTENER_DEFINE`) actually do at the linker level? Why does an observer work globally without any explicit "register" call in `main()`?

<details><summary>Answer</summary>

These macros use GCC's `__attribute__((section("...")))` to place the observer struct into a named ELF section (e.g. `.zbus.observer.packer_sub`). The linker collects all structs in any `.zbus.observer.*` section into a contiguous array in flash. ZBus's channel notification code iterates this array at runtime.

Because the array is assembled at **link time** by the linker (not at runtime by `main()`), no registration function is needed. The observers are "registered" the moment the linker sees the section and the symbol exists in a compiled .o file. This is called a *linker-generated list* or *iterable sections* pattern — the same technique Zephyr uses for shell commands (`SHELL_CMD_REGISTER`) and init functions (`SYS_INIT`).

</details>

---

**Q6.** Explain why `pb_encode()` can return `true` but the SPI frame still causes the Jetson to decode garbage. What specific check are beginners missing?

<details><summary>Answer</summary>

`pb_encode()` returns `true` to indicate "encoding completed without error." It says nothing about how many bytes were written vs. how much buffer was allocated. Beginners often write:

```c
// WRONG:
pb_decode(&in_stream, ...) where in_stream was created with sizeof(buffer)
```

The correct value to pass is `stream.bytes_written` (on the encode side), and that value must be transmitted to the decoder either in a header or as a separate field. If the decoder is given the full buffer size (e.g. 128), it tries to decode all 128 bytes, including zero-padding. Zero bytes in protobuf are valid TLV data (field 0 with wire type 0), leading to undefined decoder behavior.

The missing check: **always transmit `stream.bytes_written` in a header, never assume the full buffer is valid protobuf.**

</details>

---

**Q7.** You have a listener callback that is called 100 times per second. The callback calls `LOG_INF("accel_x=%.3f", data.accel_x)`. What is the likely failure mode, and how would you diagnose it?

<details><summary>Answer</summary>

`LOG_INF` (and `printk`) acquire an internal UART lock to format and transmit the string. This can take 1–5ms depending on UART baud rate and buffer state. Since the listener runs **inside the publisher's thread**, each publish call blocks for up to 5ms.

At 100Hz the publisher should complete each publish in <100µs. Adding a 5ms block turns 100Hz into at most 200Hz-equivalent blocking load, effectively preventing the publisher from meeting its timing budget.

**Diagnosis:**
- Add timing around `zbus_chan_pub()`: `uint32_t t0 = k_uptime_get_32();`  after the call log `t1-t0`. If it shows ~5ms instead of ~0ms, a listener is blocking.
- `zbus_chan_pub()` returns `-EAGAIN` if the timeout expires waiting for the mutex — another sign.
- The IMU thread's 100Hz k_timer will overflow semaphore count (backlogged ticks), causing the thread to process multiple ticks rapidly when it finally unblocks.

**Fix:** Remove the LOG_INF from the listener. Use a subscriber for logging.

</details>

---

**Q8.** What is the stack depth danger when calling `pb_encode()` on a `SensorFrame` that contains both `ImuData` and `WheelData` sub-messages? Why is this worse than encoding a flat message?

<details><summary>Answer</summary>

`pb_encode()` is **recursive**: for each sub-message field, it calls itself again. Each recursive invocation pushes approximately 120 bytes of pb_encode locals + a pb_ostream_t + alignment onto the stack. For a two-level nesting (SensorFrame → ImuData or WheelData):

```
pb_encode(SensorFrame)          ~120 bytes
  └─ pb_encode_submessage(ImuData)   ~80 bytes
       └─ pb_encode(ImuData)         ~120 bytes
            └─ pb_encode_fixed32()  ~16 bytes each (×7 float fields)
```

Total from pb_encode alone: ~550–750 bytes in the worst case. On top of that, the packer thread has its own function locals, and if an error path calls `LOG_ERR()`, add ~400 bytes for that.

A flat message (no sub-messages) calls pb_encode only once — one level of 120 bytes. Sub-messages multiply it by depth. With 3 levels of nesting you'd need 1KB+ for pb_encode alone.

**Practical consequence:** Threads calling pb_encode on nested messages need ≥2048 bytes of stack. The default Zephyr thread stack is 1024 bytes — insufficient.

</details>

---

**Q9.** What is the `_init_zero` idiom in nanopb and why is it mandatory? Write a code snippet showing exactly what happens if you forget it.

<details><summary>Answer</summary>

`SensorFrame_init_zero` is a generated constant that initializes the struct with: all numeric fields = 0, all `has_*` flags = `false`, all string fields = `""`. Using it guarantees a clean starting state.

If you forget it:

```c
void bad_encode_fn(void)
{
    SensorFrame frame;   // uninitialized — stack garbage!
    frame.seq = 42;      // set one field
    // frame.has_imu is now random stack bytes — might be 1 or might be another value
    // frame.imu.*  fields contain whatever was on the stack from a previous function call

    pb_ostream_t stream = pb_ostream_from_buffer(buf, sizeof(buf));
    pb_encode(&stream, SensorFrame_fields, &frame);
    // If has_imu was randomly non-zero on the stack, the encoder will attempt to encode
    // imu with random float values — looks like valid data on the Jetson,
    // but it's complete garbage that your robot's EKF will process.
}
```

This is particularly dangerous because the garbage values look like plausible IMU readings (random floats within float range), not obviously wrong values like NaN or infinity.

</details>

---

**Q10.** A colleague proposes: "When GPS loses fix, just send the last known GPS coordinates instead of zeros, so the payload size stays constant and we don't have the variable-length problem." Evaluate this proposal. Is it correct? What are its failure modes?

<details><summary>Answer</summary>

It is **incorrect** as a general solution, and introduces new failure modes:

1. **Silent position error after extended GPS outage.** If the robot moves 500 meters while GPS is down, the "last known GPS" is 500 meters stale. The Jetson's EKF fuses it as if GPS just measured the robot at the old position — injecting a large position error.

2. **`has_fix` still needs to be false.** Even if you send old coordinates, the Jetson needs to know they're stale. So you still need the `has_fix=false` flag, which is zero/default in proto3 and will be omitted — bringing you back to the same proto3 omission problem.

3. **The real fix is the `payload_length` header**, not working around proto3's omission behavior. With an explicit length header, variable-size output is no longer a problem. The Jetson always decodes exactly `payload_length` bytes, regardless of how short the payload is. GPS fields correctly show as absent/defaulted when they truly are.

</details>

---

## Section B — Spot the Bug

**Bug 1.** What is wrong with this subscriber thread?

```c
ZBUS_SUBSCRIBER_DEFINE(logger_sub, 4);

void logger_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;
    struct imu_data data;

    zbus_chan_add_obs(&imu_chan, &logger_sub, K_MSEC(5));

    while (1) {
        zbus_sub_wait(&logger_sub, &chan, K_FOREVER);
        // Read data directly from our own copy
        zbus_chan_read(&logger_sub, &data, K_NO_WAIT);
        LOG_INF("ax=%.2f", data.accel_x);
    }
}

K_THREAD_DEFINE(logger_tid, 512, logger_thread_fn, NULL, NULL, NULL, 7, 0, 0);
```

<details><summary>Answer</summary>

**THREE bugs:**

1. **Wrong argument to `zbus_chan_read()`:** The first argument should be `&imu_chan` (the channel pointer), not `&logger_sub` (the observer). You read *from the channel*, not from the subscriber.

2. **Stack too small:** `LOG_INF` with float formatting requires approximately 400 bytes. The thread does `zbus_sub_wait` (64 bytes) + `zbus_chan_read` overhead + `LOG_INF` (~400 bytes). 512 bytes is insufficient — silent stack overflow. Use 2048.

3. **At 100Hz, depth=4 is marginal:** if this thread runs any slower than 40ms (4 messages × 10ms), notifications start dropping. For a logger called 100×/sec, depth should be at least 10. With `LOG_INF` in the loop (potentially slow), drops are likely.

</details>

---

**Bug 2.** Find the bug in this listener:

```c
static void data_ready_listener(const struct zbus_channel *chan)
{
    struct imu_data data;
    zbus_chan_read(chan, &data, K_MSEC(100));

    k_sem_give(&imu_data_ready_sem);
}
ZBUS_LISTENER_DEFINE(imu_listener, data_ready_listener);
```

<details><summary>Answer</summary>

The timeout `K_MSEC(100)` on `zbus_chan_read()` inside a listener is the bug — but not for the obvious reason.

The listener is called while the channel's mutex is **already held** by the publisher. `zbus_chan_read()` also tries to acquire the channel mutex. This is a **deadlock** in most ZBus configurations: the listener tries to lock a mutex it is already holding (transitively through the publish path).

The read inside a listener should use `K_NO_WAIT`. The data was just published and the mutex is available through ZBus's internal re-entrant read semantics — but passing a timeout creates a second lock attempt that deadlocks or hangs.

Correct fix:
```c
zbus_chan_read(chan, &data, K_NO_WAIT);
```

The `k_sem_give()` itself is fine — that's a legal thing to do from a listener (it's a non-blocking ISR-safe operation).

</details>

---

**Bug 3.** What is wrong with this encoding code?

```c
uint8_t spi_buf[128];

void encode_and_send(const struct imu_data *imu)
{
    SensorFrame frame;
    frame.seq = g_seq++;
    frame.has_imu = true;
    frame.imu.accel_x = imu->accel_x;
    frame.imu.accel_y = imu->accel_y;
    frame.imu.accel_z = imu->accel_z;

    pb_ostream_t stream = pb_ostream_from_buffer(spi_buf, sizeof(spi_buf));
    pb_encode(&stream, SensorFrame_fields, &frame);

    spi_slave_set_tx_buffer(spi_buf, sizeof(spi_buf));
}
```

<details><summary>Answer</summary>

**THREE bugs:**

1. **`SensorFrame frame;` is uninitialized** — should be `SensorFrame frame = SensorFrame_init_zero;`. The `has_gps`, `has_odom`, and all unset fields contain stack garbage. The encoder may try to encode garbage sub-messages.

2. **`pb_encode()` return value is not checked.** If encoding fails (buffer too small, missing required field, corrupt struct), the error is silently ignored and `spi_buf` contains partial/corrupt data that gets sent to Jetson.

3. **`spi_slave_set_tx_buffer(spi_buf, sizeof(spi_buf))` sends 128 bytes when the payload may be much shorter.** The valid protobuf may only be 30 bytes; the remaining 98 bytes are leftover from the previous call (or uninitialized). The Jetson will decode 128 bytes, reading past the valid payload. Fix: send `stream.bytes_written` bytes (with a length header so the Jetson knows the boundary).

</details>

---

**Bug 4.** This channel definition is in a header file. What breaks?

```c
// channels.h
#ifndef CHANNELS_H
#define CHANNELS_H

#include <zephyr/zbus/zbus.h>

struct imu_data {
    float accel_x, accel_y, accel_z;
};

ZBUS_CHAN_DEFINE(                 // ← in header!
    imu_chan,
    struct imu_data,
    NULL, NULL,
    ZBUS_OBSERVERS_EMPTY,
    ZBUS_MSG_INIT(0)
);

#endif
```

<details><summary>Answer</summary>

`ZBUS_CHAN_DEFINE` **allocates memory** (channel struct, internal buffer, mutex). Placing it in a header file means every `.c` file that `#include "channels.h"` will define its own copy of `imu_chan`. The include guard `#ifndef` only prevents double-inclusion within a single translation unit — it does NOT prevent multiple translation units from each defining the symbol.

The linker will see multiple definitions of `imu_chan` and produce: `error: multiple definition of 'imu_chan'`.

**Fix:**
- `channels.h`: use `ZBUS_CHAN_DECLARE(imu_chan)` — declares as `extern`, no memory allocated
- `channels.c`: use `ZBUS_CHAN_DEFINE(imu_chan, ...)` — allocates memory exactly once

</details>

---

**Bug 5.** What does this decode code do wrong?

```c
// Jetson-side C code reading SPI
uint8_t rx_buf[128];
spi_read(rx_buf, sizeof(rx_buf));   // reads exactly 128 bytes

pb_istream_t stream = pb_istream_from_buffer(rx_buf, sizeof(rx_buf));
SensorFrame frame = SensorFrame_init_zero;
bool ok = pb_decode(&stream, SensorFrame_fields, &frame);
```

<details><summary>Answer</summary>

`pb_istream_from_buffer(rx_buf, sizeof(rx_buf))` tells the decoder to parse **all 128 bytes** as valid protobuf. But the valid payload is only a subset — the remainder is zero-padding from the `encode_sensor_frame()` zero-fill step (or stale DMA bytes).

The decoder will encounter the zero bytes and attempt to interpret them as protobuf TLV data. Zero bytes in protobuf encoding mean field 0 with wire type 0 — an invalid field tag that causes decoder errors (or, if the decoder is lenient, spurious zero values overwriting correctly decoded fields).

**Fix:** Read the `payload_length` from the header bytes first:
```c
uint16_t payload_len = (rx_buf[2] << 8) | rx_buf[3];
pb_istream_t stream = pb_istream_from_buffer(rx_buf + 4, payload_len);
```

</details>

---

**Bug 6.** This code uses a ZBus listener to handle incoming SPI data requests. Find the problem:

```c
static void spi_request_listener_cb(const struct zbus_channel *chan)
{
    struct spi_request req;
    zbus_chan_read(chan, &req, K_NO_WAIT);

    // Prepare the response buffer
    uint8_t response[64];
    encode_response(&req, response, sizeof(response));

    // Write to SPI peripheral register
    spi_write_blocking(SPI1, response, sizeof(response));
}
ZBUS_LISTENER_DEFINE(spi_request_handler, spi_request_listener_cb);
```

<details><summary>Answer</summary>

`spi_write_blocking()` is a **blocking** call — it waits for the SPI peripheral to complete the entire transfer (potentially milliseconds for 64 bytes at low baud rate). A listener runs inside the **publisher's thread context**. Calling a blocking function from a listener blocks the publisher for the entire SPI transfer duration.

Additionally, it may call functions that try to sleep (`k_msleep`) or take mutexes — both of which are illegal in certain listener contexts (similar to ISR restrictions).

**Fix:** The listener should only set a flag or give a semaphore. A separate thread reads the flag and does the actual SPI write:
```c
static void spi_request_listener_cb(const struct zbus_channel *chan)
{
    // Cache the request (fast copy)
    zbus_chan_read(chan, &g_pending_request, K_NO_WAIT);
    k_sem_give(&spi_work_sem);   // fast, non-blocking, ISR-safe
}
// A separate thread waits on spi_work_sem and does the actual SPI write
```

</details>

---

**Bug 7.** What is the versioning bug in this `.proto` change?

```protobuf
// sensor_frame.proto v1
message ImuData {
    float  accel_x      = 1;
    float  accel_y      = 2;
    float  accel_z      = 3;
    uint64 timestamp_us = 4;
}

// sensor_frame.proto v2 — new engineer added temperature
message ImuData {
    float  accel_x      = 1;
    float  accel_y      = 2;
    float  temperature  = 3;   // ← new field placed at old accel_z's number
    float  accel_z      = 4;   // ← accel_z moved to field 4
    uint64 timestamp_us = 5;   // ← timestamp_us also renumbered
}
```

<details><summary>Answer</summary>

**Critical protobuf versioning violation:** field numbers were **reused and renumbered**.

Field numbers are the permanent wire identity of each field. An old decoder (Jetson running v1 software) receiving a v2 frame will:
- Read field 3 → decode it as `accel_z` (old meaning) but parse the bytes as `float temperature` value → `accel_z` gets whatever temperature was
- Read field 4 → decode it as... there was no field 4 in v1 → unknown field, silently skipped → `accel_z` stays at zero

The Jetson running v1 software now has `accel_z=temperature_value` and `timestamp_us=accel_z_value`. The EKF receives completely wrong data with no error.

**Rules:**
1. Never reuse a field number
2. Never renumber an existing field
3. New fields get new numbers appended at the end
4. Retired fields use `reserved 3;` to prevent reuse

Correct v2:
```protobuf
message ImuData {
    float  accel_x      = 1;
    float  accel_y      = 2;
    float  accel_z      = 3;   // unchanged
    uint64 timestamp_us = 4;   // unchanged
    float  temperature  = 5;   // NEW — gets next available number
}
```

</details>

---

**Bug 8.** This code is in the 100Hz publisher thread. What is the race condition?

```c
static struct imu_data g_imu_shared;   // global — packer reads this

void imu_thread_fn(void *a, void *b, void *c)
{
    struct imu_data local;

    while (1) {
        read_icm42688(&local);
        local.timestamp_us = k_ticks_to_us_near64(k_uptime_ticks());

        // Publish to ZBus (safe, atomic copy)
        zbus_chan_pub(&imu_chan, &local, K_MSEC(1));

        // ALSO update the global for direct access by legacy code
        g_imu_shared = local;   // struct assignment

        k_msleep(10);
    }
}
```

<details><summary>Answer</summary>

`g_imu_shared = local` is a **struct assignment**, which the compiler implements as multiple 32-bit store instructions (one per field). This is NOT atomic. The packer thread (or any other thread) can read `g_imu_shared` between two of these stores, getting a partially-updated struct — exactly the race condition ZBus was supposed to prevent.

The code has carefully used `zbus_chan_pub()` (safe) for the primary path, but then created a second unsafe path by maintaining a global struct. Any consumer of `g_imu_shared` bypasses ZBus's mutex protection entirely.

**Fix:** Remove `g_imu_shared`. All consumers must use `zbus_chan_read()`. If legacy code needs direct access and cannot be modified, protect the global with a `struct k_mutex` on every access.

</details>

---

## Section C — Fill in the Blank

**C1.** Complete the `ZBUS_CHAN_DEFINE` call for a GPS channel that uses `struct gps_data`, has no validator, is initially zeroed, and has no compile-time observers (observers will be added at runtime):

```c
ZBUS_CHAN_DEFINE(
    _________,           // channel name
    _________,           // message type
    _________,           // validator (none)
    _________,           // user_data (none)
    _________,           // observers
    _________            // initial value
);
```

<details><summary>Answer</summary>

```c
ZBUS_CHAN_DEFINE(
    gps_chan,
    struct gps_data,
    NULL,
    NULL,
    ZBUS_OBSERVERS_EMPTY,
    ZBUS_MSG_INIT(0)
);
```

</details>

---

**C2.** Complete the subscriber thread pattern. The subscriber should wait on `imu_chan` notifications with no timeout and read the data when notified:

```c
ZBUS_SUBSCRIBER_DEFINE(analysis_sub, _____);   // queue depth for 100Hz with 50ms tolerance

void analysis_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;
    struct imu_data data;

    // Register as observer of imu_chan
    _____(_____, _____, K_MSEC(5));

    while (1) {
        int rc = _____(_____, _____, _____);   // block until notification
        if (rc != 0) continue;

        if (chan == _____) {
            _____(_____, _____, K_NO_WAIT);    // read latest
            run_analysis(&data);
        }
    }
}
```

<details><summary>Answer</summary>

```c
ZBUS_SUBSCRIBER_DEFINE(analysis_sub, 5);   // 50ms/10ms = 5 slots

void analysis_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;
    struct imu_data data;

    zbus_chan_add_obs(&imu_chan, &analysis_sub, K_MSEC(5));

    while (1) {
        int rc = zbus_sub_wait(&analysis_sub, &chan, K_FOREVER);
        if (rc != 0) continue;

        if (chan == &imu_chan) {
            zbus_chan_read(&imu_chan, &data, K_NO_WAIT);
            run_analysis(&data);
        }
    }
}
```

</details>

---

**C3.** Fill in the `encode_sensor_frame()` function body for the encode stream setup and the mandatory checks:

```c
uint16_t encode_sensor_frame(SensorFrame *frame, uint8_t *buf, size_t buf_size)
{
    // 1. Set up output stream
    pb_ostream_t stream = _________________________;

    // 2. Encode
    bool ok = _________________________;

    // 3. Check for error and log it
    if (!ok) {
        LOG_ERR("pb_encode failed: %s", _________);
        return 0;
    }

    // 4. Return the ACTUAL bytes written (not sizeof(buf))
    return (uint16_t)_____________;
}
```

<details><summary>Answer</summary>

```c
uint16_t encode_sensor_frame(SensorFrame *frame, uint8_t *buf, size_t buf_size)
{
    pb_ostream_t stream = pb_ostream_from_buffer(buf, buf_size);

    bool ok = pb_encode(&stream, SensorFrame_fields, frame);

    if (!ok) {
        LOG_ERR("pb_encode failed: %s", PB_GET_ERROR(&stream));
        return 0;
    }

    return (uint16_t)stream.bytes_written;
}
```

</details>

---

**C4.** Complete the SPI framing header write. The magic bytes are `0xDE` `0xAD`. `payload_len` is 16-bit big-endian:

```c
void write_frame_header(uint8_t *frame_buf, uint16_t payload_len)
{
    frame_buf[0] = _____;
    frame_buf[1] = _____;
    frame_buf[2] = _____;   // high byte of payload_len
    frame_buf[3] = _____;   // low byte of payload_len
}

// On the Jetson receiver, extract the header:
bool parse_frame_header(const uint8_t *buf, uint16_t *out_len)
{
    if (buf[0] != _____ || buf[1] != _____) {
        return false;  // desync
    }
    *out_len = (____)                ;  // reconstruct uint16_t
    return true;
}
```

<details><summary>Answer</summary>

```c
void write_frame_header(uint8_t *frame_buf, uint16_t payload_len)
{
    frame_buf[0] = 0xDE;
    frame_buf[1] = 0xAD;
    frame_buf[2] = (uint8_t)(payload_len >> 8);
    frame_buf[3] = (uint8_t)(payload_len & 0xFF);
}

bool parse_frame_header(const uint8_t *buf, uint16_t *out_len)
{
    if (buf[0] != 0xDE || buf[1] != 0xAD) {
        return false;
    }
    *out_len = ((uint16_t)buf[2] << 8) | buf[3];
    return true;
}
```

</details>

---

**C5.** Fill in the `.options` file entry and the `.proto` message for a struct that includes a robot name (max 32 chars including null) and a status string (max 16 chars including null):

```protobuf
// status.proto
message RobotStatus {
    uint32 robot_id  = 1;
    string _______   = 2;   // robot name
    string _______   = 3;   // status
    uint64 uptime_ms = 4;
}
```

```ini
# status.options
RobotStatus._______   max_size:____
RobotStatus._______   max_size:____
```

<details><summary>Answer</summary>

```protobuf
message RobotStatus {
    uint32 robot_id  = 1;
    string name      = 2;
    string status    = 3;
    uint64 uptime_ms = 4;
}
```

```ini
RobotStatus.name    max_size:32
RobotStatus.status  max_size:16
```

</details>

---

**C6.** A thread for the packer is defined below. Fill in the missing values based on these requirements: must tolerate 20ms of ZBus backlog at 100Hz; must accommodate pb_encode + LOG_ERR on stack; should run at lower priority than the IMU thread (priority 3) but higher than idle:

```c
K_THREAD_DEFINE(
    packer_tid,
    _____,              // stack size
    packer_thread_fn,
    NULL, NULL, NULL,
    _____,              // priority
    0,
    0
);

ZBUS_SUBSCRIBER_DEFINE(packer_sub, _____);   // queue depth
```

<details><summary>Answer</summary>

```c
K_THREAD_DEFINE(
    packer_tid,
    2048,              // 2048: pb_encode ~750 bytes + LOG_ERR ~400 bytes + margin
    packer_thread_fn,
    NULL, NULL, NULL,
    4,                 // priority 4: lower than IMU (3), higher than default idle (14)
    0,
    0
);

ZBUS_SUBSCRIBER_DEFINE(packer_sub, 2);   // 20ms / 10ms = 2 notification slots
```

Note: queue depth 2 is the minimum for 20ms tolerance. In practice use 4–10 for safety against brief scheduler delays.

</details>

---

## Section D — Lab Tasks

**Lab 1: ZBus Silent Drop Detection**

**Goal:** Reproduce and measure the ZBus silent drop behavior.

**Setup:**
1. Create an IMU simulator thread that publishes to `imu_chan` every 10ms, with a `seq` field that increments by 1 each publish.
2. Create a subscriber thread with queue depth **1** and simulate it being "slow" by adding `k_msleep(25)` inside the loop.
3. On the Jetson (or via UART log), print `delta_seq = current_seq - prev_seq` for every received message.

**Verification criteria:**
- Without the `k_msleep(25)`: `delta_seq` should always be 1. If you see any values > 1, your test setup is wrong.
- With the `k_msleep(25)`: `delta_seq` should frequently be 2–3. Log these as WRN.
- Add `zbus_obs_get_chan_drop_cnt()` to a shell command and verify the drop count matches your `delta_seq > 1` count.
- Increase queue depth from 1 to 4 to 10 and observe how the drop number changes.

**Expected learning:** You'll see that identical channel values for N consecutive reads (when `delta_seq > 1`) is the signature of drops — not sensor stiction.

---

**Lab 2: Variable-Length Encoding Edge Case**

**Goal:** Observe the GPS zero-omission problem and verify the payload_length fix.

**Setup:**
1. Create a `SensorFrame` with full GPS data (latitude=35.69, longitude=139.69, has_fix=true).
2. Encode it and record `stream.bytes_written` — call this `size_with_fix`.
3. Set all GPS fields to zero (has_fix=false, lat=0, lon=0, alt=0).
4. Encode it and record `stream.bytes_written` — call this `size_without_fix`.

**Verification criteria:**
- `size_with_fix > size_without_fix` (GPS fields were omitted = smaller output). The difference should be ≥10 bytes.
- Take the "with fix" encoded bytes. Put them in a 128-byte buffer (zeros for remainder). Decode using `sizeof(128_byte_buffer)` — observe what fields come out wrong.
- Now decode using `size_with_fix` only — verify all fields decode correctly.
- Repeat with the "without fix" payload.

**Expected learning:** The difference in encoded size confirms proto3 omits zero fields. The importance of `payload_length` becomes concrete — you can see *exactly* how many bytes are valid.

---

**Lab 3: Stack High-Water Mark for pb_encode**

**Goal:** Measure the actual stack impact of pb_encode with nested sub-messages.

**Setup:**
1. Create a thread with a 1024-byte stack that calls `pb_encode()` on a `SensorFrame` with both `ImuData` and `WheelData` populated.
2. Enable `CONFIG_STACK_SENTINEL=y` and `CONFIG_THREAD_STACK_INFO=y`.
3. After 100 encode calls, call `k_thread_stack_space_get()` and log the result.

**Verification criteria:**
- With 1024-byte stack + `CONFIG_STACK_SENTINEL=y`: expect to see a stack sentinel violation panic (confirms the stack is too small).
- With 2048-byte stack: `k_thread_stack_space_get()` should show ≥200 bytes unused.
- With 4096-byte stack: verify unused space doubles — confirms linear relationship.
- Add `LOG_ERR()` to the error path (even if never triggered) and measure stack increase.

**Expected learning:** pb_encode stack usage is real and measurable. Using `k_thread_stack_space_get()` is the correct way to size threads rather than guessing.

---

**Lab 4: Round-Trip Test on Linux Before Porting**

**Goal:** Validate the full encode→decode pipeline on a Linux host before touching the STM32.

**Setup:**
1. Write `test/round_trip.c` (see study notes section 2.7) for your actual `SensorFrame` schema.
2. Compile with: `gcc round_trip.c sensor_frame.pb.c -Inanopb -lnanopb -o round_trip`
3. Add test cases for ALL edge cases:
   - Normal: all fields populated
   - GPS no fix: GPS fields all zero, `has_fix=false`
   - Minimal: only `seq` set, no sub-messages
   - Max values: float `FLT_MAX`, `timestamp_us` = `UINT64_MAX / 2`

**Verification criteria:**
- All test cases pass assert checks without memory errors (run under valgrind: `valgrind ./round_trip`).
- Print hex dumps for each test case. Label them and keep them. These are your reference for debugging STM32-side output.
- For the "GPS no fix" case: confirm that encoded size is smaller than "normal" case.
- The round-trip test must pass before you write a single line of firmware packer code.

**Expected learning:** Bugs found on Linux in 5 minutes would cost 2 hours on STM32 (flash-debug cycle). This is the "test on host first" discipline.

---

## Section E — Design Challenges

**Design 1: Multi-sensor ZBus architecture**

You're adding three new sensors to the robot: a LIDAR (10Hz), a barometer (50Hz), and a battery monitor (1Hz). Currently the packer thread subscribes to a single `imu_chan` at 100Hz.

**Question:** Design the ZBus channel + observer architecture for the complete multi-sensor system. Consider:
- Should the packer subscribe to all channels individually, or should there be a "data aggregator" thread?
- How do you handle the 100× rate mismatch between LIDAR (10Hz) and IMU (100Hz)?
- The battery monitor data is critical — what happens if its ZBus notification is dropped?
- What queue depths do you choose for each sensor's subscriber?

Write a table showing: sensor → channel → subscriber/listener → thread priority → queue depth, and justify each choice.

<details><summary>Answer (indicative)</summary>

**Recommended architecture: separate channels, aggregator thread**

| Sensor | Channel | Observer type | Observer | Priority | Queue depth | Rationale |
|---|---|---|---|---|---|---|
| IMU 100Hz | `imu_chan` | Subscriber | `aggregator_sub` | 4 | 10 | 100ms burst tolerance |
| LIDAR 10Hz | `lidar_chan` | Subscriber | `aggregator_sub` | 4 | 3 | 300ms tolerance; LIDAR updates are slow |
| Barometer 50Hz | `baro_chan` | Subscriber | `aggregator_sub` | 4 | 5 | 100ms tolerance |
| Battery 1Hz | `batt_chan` | Listener | `batt_critical_cb` | N/A | 0 (sync) | Critical: must not be dropped. Listener fires async with zero latency. Callback only sets a flag. |
| Battery 1Hz | `batt_chan` | Subscriber | `batt_logger_sub` | 12 | 2 | Logging is low priority; can miss a beat |

**Aggregator thread:** waits on `zbus_sub_wait()` for any of the 3 subscribed channels. Caches the latest value from each. When IMU fires (100Hz), encodes a `SensorFrame` with the cached latest values for all sensors. This ensures the SPI frame runs at 100Hz but always includes the most recent data from slower sensors.

**Battery subscriber depth of 2:** at 1Hz, you'd need to be slow by > 2 seconds to drop. Acceptable. But the listener on the same channel gives a zero-latency warning if voltage goes critical.

</details>

---

**Design 2: Detecting ZBus drops in production**

Your robot runs at 100Hz for 8-hour shifts. Occasionally, operators report that the robot "acts weird" for 1–2 seconds, then recovers. Looking at the rosbag, you see 3–4 consecutive identical sensor frames.

**Question:** Design a monitoring system to catch this in production.

1. What fields must be added to `SensorFrame` to make drops detectable by the Jetson?
2. What Zephyr shell commands or metrics would you expose for the robot operator?
3. If drops exceed 5 in any 1-second window, you want an automatic alert. Design the alert path: which thread raises it, how it travels (ZBus? direct log? CAN?), and who consumes it.
4. How would you distinguish "ZBus drops" from "packer thread starvation" from "Jetson SPI desync"?

Write a 10-line design summary. There is no single correct answer — scoring is based on identifying the right questions and showing the tradeoffs.

<details><summary>Answer (indicative)</summary>

**Design summary:**

1. **Add `seq` (uint32) and `drop_count` (uint32) to SensorFrame.** `seq` increments on every encode. `drop_count` is read from `zbus_obs_get_chan_drop_cnt(&packer_sub)` and included in each frame. Jetson detects `delta_seq > 1` as a drop event.

2. **Expose via Zephyr shell:** `zbus stats` command printing per-channel drop counts. `packer stats` command printing encode count, last encode latency, stack high-water mark. Log these to UART every 60 seconds at INFO level.

3. **Alert path:** A monitoring thread reads `delta_seq` on the Jetson side. If it counts >5 drops in 1000ms, it publishes a `/robot/health` ROS2 topic with `DROP_STORM` status. The robot's behavior arbiter subscribes to this topic and switches to a degraded-mode motion planner. Additionally, the STM32 shell command is exposed over CAN as a diagnostic frame, so operators can query even if SPI is flaky.

4. **Distinguishing causes:**
   - ZBus drops: `zbus_obs_get_chan_drop_cnt()` > 0 on STM32 side. Jetson `delta_seq > 1`. Frame timestamps consecutive (STM32 kept going, just packer was slow).
   - Packer starvation: encode count stops incrementing. `packer stats` shows last encode timestamp is old.
   - Jetson SPI desync: magic byte check fails. Jetson sees `buf[0] != 0xAB`. Desync counter increments.

</details>

---

**Design 3: GPS fix-loss forward compatibility**

Future firmware (v3) will add a `gps_accuracy_m` float field (field number 8) to `GpsData`. The current firmware on installed robots (v2) does not have this field. Jetson software will be updated before all robot firmwares are.

**Question:**
1. What happens when updated Jetson v3 software receives a frame from old firmware v2 (without field 8)?
2. What happens when v2 Jetson software receives a frame from new firmware v3 (with field 8)?
3. Is there any code change needed on either side to make this safe?
4. What test must you run in the CI pipeline to verify forward/backward compatibility? Describe the minimal test fixture.

<details><summary>Answer (indicative)</summary>

1. **Jetson v3 receives from firmware v2:** Field 8 (`gps_accuracy_m`) is absent in the encoded frame. The v3 decoder defaults the missing field to 0.0. The Jetson code must handle `gps_accuracy_m == 0.0` as "not available" — not as "100% accuracy." This is a code convention issue: document that 0.0 means "field absent/unknown."

2. **Jetson v2 receives from firmware v3:** Field 8 is present in the frame. The v2 decoder sees an unknown field tag (8). Proto3 decoders silently skip unknown fields. The rest of the message decodes correctly. No crash, no error.

3. **Code changes needed:** Only on the Jetson v3 side — add logic to treat `gps_accuracy_m == 0.0` as "unknown" rather than "zero meters accuracy." No changes needed on the firmware v2 side — it simply doesn't encode field 8.

4. **CI test fixture:**
   - Encode a `GpsData` using the v3 `.pb.c` (with field 8 set to 5.0).
   - Decode using the v2 `.pb.c` (without field 8 in its field table). Verify: existing fields decode identically, no crash.
   - Encode a `GpsData` using the v2 `.pb.c` (without field 8).
   - Decode using the v3 `.pb.c`. Verify: `gps_accuracy_m == 0.0` (default), existing fields correct.
   - This can be automated in the `round_trip.c` test from Lab 4 by keeping old and new `.pb.c` files and linking them separately.

</details>
