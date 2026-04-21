# ZBus + nanopb — Study Notes
### Project 5: Sensor Packer Pipeline (ZBus → nanopb → SPI)
**Hardware:** STM32 Nucleo-H743ZI2 · ICM-42688-P · Jetson Orin · Zephyr RTOS

> **Electronics prerequisite:** Understanding the SPI frame framing (header + length + CRC) assumes you've read:
> - SPI protocol framing → `electronics/05-spi-deep-dive.md` (why raw SPI has no message boundaries)
> - Serialization concept → `electronics/03-opamps-adc-sampling.md` § Serializer/Deserializer

---

## PART 1 — ELI15 Concept Explanations

---

### 1.1 Why you need ZBus — what happens without it

**The problem: shared state in a concurrent world**

Imagine a whiteboard in an office. Three employees write on it at different times. Now imagine two of them try to write on it simultaneously — you get a half-written mess no one can read.

In an embedded system with multiple threads, a global struct is that whiteboard. Thread A is reading `imu.accel_x` right as Thread B is writing a new measurement. The read gets half the old value and half the new value. Not garbage-looking garbage — plausible-looking garbage. The worst kind.

Here is *exactly* what can go wrong on a 32-bit ARM without protection:

```c
// A 24-byte struct on ARM Cortex-M
struct imu_data {
    float accel_x;    // 4 bytes
    float accel_y;    // 4 bytes
    float accel_z;    // 4 bytes
    float gyro_x;     // 4 bytes
    float gyro_y;     // 4 bytes
    float gyro_z;     // 4 bytes
};

struct imu_data g_imu;           // global — shared between threads

// Thread A writes (runs every 10ms, IRQ-driven)
void imu_irq_handler(void) {
    g_imu.accel_x = read_reg(ACCEL_X);  // 4 bytes
    g_imu.accel_y = read_reg(ACCEL_Y);  // 4 bytes
    // ← Preemption can happen HERE, between fields
    g_imu.accel_z = read_reg(ACCEL_Z);  // 4 bytes
}

// Thread B reads (packer thread)
void pack_fn(void) {
    // If preempted mid-write above, accel_z is still from the PREVIOUS measurement
    // accel_x and accel_y are new, accel_z is old
    // The struct is internally inconsistent
    float ax = g_imu.accel_x;    // new value ✓
    float az = g_imu.accel_z;    // OLD value from 10ms ago ✗
}
```

**The "just use a mutex" trap:**

You could fix this with a mutex, but now you've created two new problems:
1. Every reader and every writer must remember to lock/unlock. Forget once = bug.
2. The writer (often the IMU thread needing to be fast) now blocks on *every single write* waiting for all readers to finish. If any reader holds the lock too long, you miss samples.

ZBus solves both. It wraps the mutex internally so you can never forget it, and its model is *copy-on-publish* — the channel holds one latest copy, and ZBus handles the locking.

**The signal-that-also-carries-data problem:**

In ROS you're used to: "subscribe to `/imu/data`" — you get both the *notification* that data arrived AND the data itself, bundled. In bare RTOS code without ZBus, there's a mismatch. A semaphore gives you the notification. A mutex-protected struct gives you the data. You need both, and you need them to stay in sync. ZBus provides both together.

---

### 1.2 The pub/sub mental model vs calling a function — decoupling

**Calling a function: tight coupling**

When Module A calls `packer_handle_imu(&data)` directly:
- A must know B exists
- A must know B's function signature
- If B changes, A must change too
- A blocks while B runs
- You cannot add a third consumer of IMU data without modifying A

This is like a food vendor who personally hand-delivers food to only the three customers he knows. To add a fourth customer he has to change his route. If customer 2 is slow eating, he waits.

**Pub/sub: loose coupling**

ZBus is a bulletin board. The IMU thread pins a note on the board (`zbus_chan_pub`). It walks away immediately and never looks back. It doesn't know — or care — who reads the note.

The packer thread, the logger thread, and a new debug-visualizer thread can ALL read the same bulletin board entry, independently. The IMU thread's code is identical whether 0, 1, or 10 consumers exist.

```
            ┌─────────────────────────────────┐
            │         ZBus Channel             │
            │                                  │
    IMU     │  pub()    ┌─────────────────┐    │
    thread ──────────► │  latest imu_data │    │
            │           └─────────────────┘    │
            │                  │               │
            │         ┌────────┴──────────────────────────┐
            │         │        │               │           │
            │      packer   logger         debugger   future_thread
            │      thread   thread         thread     (doesn't exist yet)
            └─────────────────────────────────┘
```

The IMU thread has zero dependencies on any downstream consumer. Add or remove consumers → zero changes to the publisher.

---

### 1.3 ZBus channel vs subscriber vs listener — three notification mechanisms

**Channel — the storage**

The channel is just the data slot. Think of it as a lockable cubby hole. Anyone can peek in (read) or replace what's inside (publish), as long as they use the provided lock. The channel doesn't actively *do* anything — it just holds the latest value.

**Subscriber — asynchronous, queued, runs in its own thread**

Imagine you subscribe to a YouTube channel. When a new video is uploaded you get a notification in your inbox. You read it when you have time. If 5 videos are uploaded while you're offline, you have 5 pending notifications — up to your inbox size. If your inbox is full (subscriber queue depth reached), the oldest notification is dropped.

```
                   ┌──────────────┐
  Channel pub() ──► Notification  │= message queue (depth 4)
                   │ queue       ├──► subscriber thread reads when it wakes
                   └──────────────┘
```

Use a subscriber when your processing might take longer than the publish period, or when you want the processing to run in a dedicated thread.

**Listener — synchronous, no queue, runs in the publisher's thread**

A listener is not a separate inbox — it's a callback that gets called *inside* the publisher's thread, *right now*, before `zbus_chan_pub()` even returns. Like a boss who says "whenever an order comes in, immediately tell Sophie" — Sophie stops what she's doing, does the thing, then the order process continues.

```
  Publisher                           Listener callback
  calls pub()
       │
       ▼
  lock channel
  copy data in
  notify observers ──────────────► imu_log_cb() runs HERE (in publisher's thread!)
  ◄──────────────────────────────── callback returns
  unlock channel
  pub() returns to publisher
```

**This has a critical consequence:** if your listener does ANYTHING slow — an I2C write, a flash operation, even a heavy calculation — you are blocking the publisher. At 100Hz, blocking for 5ms means you've just broken your timing budget.

**Quick decision guide:**

| Need | Use |
|-|-|
| Record data in a file or database | Subscriber |
| Compute a running average | Subscriber |
| Set a flag so another thread wakes | Listener |
| Increment a packet counter | Listener |
| Encode protobuf and write to SPI | Subscriber |
| Blink a status LED | Listener |

---

### 1.4 What ZBUS_OBSERVER_DEFINE does under the hood

**The compile-time list trick**

ZBus uses a C99 feature called *linker sections* to build its observer registry at compile time with zero runtime overhead. Understanding this prevents confusion about why things work even without any explicit "register" call.

When you write:
```c
ZBUS_SUBSCRIBER_DEFINE(imu_sub, 4);
```

The macro expands to roughly:
```c
// A k_msgq is defined for the notification queue
K_MSGQ_DEFINE(_imu_sub_queue, sizeof(void *), 4, 4);

// An observer struct is created and placed in a special linker section
const struct zbus_observer imu_sub
    __attribute__((section(".zbus.observer.imu_sub"))) = {
    .name = "imu_sub",
    .type = ZBUS_OBSERVER_SUBSCRIBER_TYPE,
    .data.queue = &_imu_sub_queue,
    .enabled = ATOMIC_INIT(1),
};
```

That `__attribute__((section(...)))` places the struct into a named region of flash memory. The linker collects *all* structs placed in any `.zbus.observer.*` section and arranges them consecutively in flash. Zephyr's ZBus code then iterates this array at notification time.

**What does this mean for you?**
- You never need to register observers in an init function — the linker does it
- The list is fully known at compile time — no dynamic allocation occurs when `zbus_chan_pub` runs
- Adding a new observer nowhere near the channel definition works because the linker sees all sections

When you add `ZBUS_CHAN_ADD_OBS(imu_chan, imu_sub, 3)` with priority 3, ZBus records in the channel's observer list that `imu_sub` should be notified. This can be done at compile time (via `ZBUS_OBSERVERS(...)` in `ZBUS_CHAN_DEFINE`) or at runtime (via `ZBUS_CHAN_ADD_OBS` with `CONFIG_ZBUS_RUNTIME_OBSERVERS=y`).

---

### 1.5 Thread-safety guarantees ZBus provides — and what it does NOT protect

**What ZBus guarantees:**

Every `zbus_chan_pub()` and `zbus_chan_read()` takes a mutex on the channel before touching its data. This means:
- Two threads can never simultaneously write the channel — one blocks on the mutex
- A reader and a writer cannot race — one blocks
- The data you read is always a complete, consistent struct (all fields from the same publish call)

**What ZBus does NOT protect:**

The mutex only covers the *channel's internal buffer copy*. The struct you pass TO `pub()` or receive FROM `read()` is your local copy — ZBus copies it in or out atomically, but after the copy, the mutex is released. What you do with your local copy is your problem.

```c
// This is SAFE — ZBus copies atomically
struct imu_data local_copy;
zbus_chan_read(&imu_chan, &local_copy, K_MSEC(1));  // mutex held during copy
// local_copy is now consistent — all fields from one publish call
float ax = local_copy.accel_x;  // safe

// This would be UNSAFE — but only if you shared local_copy with another thread:
// Don't pass &local_copy to another thread and read while they mutate it
```

**The "only latest value" contract:**

ZBus channels are NOT a ring buffer. Publishing twice before anyone reads means the first value is silently overwritten. This is correct behavior for sensor data (you want the most recent reading), but surprising if you're used to a message queue model. If you need every message delivered, use a subscriber's queue, and size it for the burst rate you expect.

---

### 1.6 What protobuf is — binary serialization vs JSON/text

**The problem with raw C structs over a wire:**

A `struct imu_data` is just bytes in memory. If you `memcpy` it and send it over SPI, the receiver must know:
- Exactly how large the struct is
- The byte order of each field (little-endian vs big-endian)
- The exact compiler padding between fields (depends on compiler, architecture, and flags)
- What each offset means

Change any compiler flag on either side → corruption. Add a field to the struct → every receiver must be recompiled simulataneously.

**Why not JSON?**

JSON is human-readable but expensive: `{"accel_x": 1.234567}` is 18 bytes to encode one float. On a microcontroller at 100Hz, this means parsing and formatting strings 100 times per second. Text serialization also requires a dynamic allocator or a very large stack for `sprintf`.

**What protobuf does instead:**

Protobuf encodes data as a binary stream of Tag-Value pairs. Each field has a numeric ID (defined in the `.proto` file). The receiver uses the IDs — not field positions — to decode. This means:

```
accel_x = 1.234567 encoded as:
  [0x0D]  ← tag byte: field 1, wire type 5 (32-bit)
  [3F 9E F9 DB]  ← IEEE 754 little-endian for 1.234567f
  Total: 5 bytes

vs

"accel_x": 1.234567  ← 18 bytes as text
```

**The field-number-based encoding** is the key to safe versioning. Field 1 always means `accel_x` — not "the first field in memory." So:
- Old firmware sends fields 1–7. New firmware adds field 8 for temperature.
- Old Jetson software receives field 8 → it sees an unknown tag → skips it silently. No crash.
- New Jetson software receives from old firmware → field 8 is missing → decoded as 0 (numeric default). No crash.

You can evolve the protocol independently on each side.

---

### 1.7 Why proto3 omits default (zero) values — and the SPI frame size problem

**Proto3's space-saving rule:**

In proto3, any field that equals its default value is **not encoded**. The default for all numeric types is zero. The default for strings is empty string.

This seems like a great idea — why transmit zeros? A `SensorFrame` with only `seq=5` and all sensor data zero would encode to just 2 bytes (the seq tag + value), instead of 64 bytes.

**The SPI framing problem this creates:**

SPI is a fixed-frame protocol in your system design. You have a DMA buffer of, say, 128 bytes. The Jetson expects every transfer to carry exactly 128 bytes. But `pb_encode()` produces output of variable size depending on the data:

```
GPS has fix:  SensorFrame with GPS lat/lon/alt all non-zero
              → pb_encode writes ~90 bytes to buffer

GPS loses fix: SensorFrame with GPS lat=0, lon=0, alt=0 (all default)
               → GPS sub-message fields are ALL OMITTED
               → pb_encode writes only ~64 bytes to buffer

The Jetson still reads 128 bytes of SPI (that's what the DMA transfer is sized for).
The last 64 bytes of the buffer are from the PREVIOUS transfer (DMA buffer was not cleared).
The Jetson runs pb_decode on 128 bytes where only 64 are valid → garbage decoded fields.
```

This is not a theoretical problem — it appeared in the OKS sensorbar investigation (#98301). Identical values across 3–4 consecutive 10ms windows is the symptom: the Jetson is reading stale bytes from the DMA buffer.

**The fix: explicit `payload_length`:**

Every SPI frame MUST include a length prefix. The packer thread writes the encoded byte count into the first 2 bytes (or into a dedicated header field). The Jetson reads the length first, then decodes only that many bytes:

```
Frame layout in the 128-byte DMA buffer:
  Bytes 0-1:   payload_length (uint16_t, little-endian)  ← always present
  Bytes 2-N:   protobuf payload (variable, exactly payload_length bytes)
  Bytes N+1-127: unused / zero padding
```

This is why the `.proto` for `SensorFrame` always has a `payload_length` field — it exists outside protobuf encoding so the framing layer can determine decode boundaries before the protobuf decoder even starts.

---

### 1.8 What nanopb is — heap-free, static allocation only

**Standard protobuf libraries (protobuf-c, Google's C++ impl) use malloc:**

```c
// Standard protobuf-c — NOT suitable for embedded:
SensorFrame *frame = sensor_frame__unpack(NULL, len, buf);
// ^ calls malloc() internally
// On STM32H7 with 1MB flash and Zephyr kernel, you might not have a heap configured at all
sensor_frame__free_unpacked(frame, NULL);  // must free or you leak
```

On a microcontroller, `malloc` has three problems:
1. The heap may not exist or may be too small
2. Malloc has non-deterministic latency (depends on heap fragmentation)
3. Free-then-malloc cycles cause fragmentation over hours — your robot crashes at 3am

**nanopb allocates everything statically:**

nanopb generates C structs where all arrays have fixed maximum sizes. You allocate the struct on the stack or as a global — no `malloc` is ever called during encode or decode. The codec is purely deterministic — same input always takes the same path and the same time.

```c
// nanopb — safe for embedded:
SensorFrame frame = SensorFrame_init_zero;  // stack allocation, fixed size
pb_ostream_t stream = pb_ostream_from_buffer(buf, sizeof(buf));  // static buffer
pb_encode(&stream, SensorFrame_fields, &frame);  // zero malloc calls
```

The tradeoff: you must declare max sizes for variable-length fields (strings, byte arrays) at code-generation time using a `.options` file. If runtime data exceeds that max, encode fails — but predictably, not with a heap corruption.

---

### 1.9 The .proto file + .options file combo

**The .proto file defines the logical schema:**

```protobuf
message GpsData {
    double latitude    = 1;
    double longitude   = 2;
    float  altitude_m  = 3;
    string fix_type    = 4;   // "no_fix", "2D", "3D"
    uint32 satellites  = 5;
}
```

The problem: `string fix_type = 4` is a variable-length string. nanopb cannot generate a struct without knowing the maximum length. nanopb can guess, but the guess is wrong.

**The .options file sets nanopb-specific constraints:**

Create a file with the same name as the .proto but `.options` extension:

```ini
# sensor_frame.options
GpsData.fix_type    max_size:8    # "no_fix" is 6 chars; max 7 + null = 8
```

Without this:
```
error: Field 'fix_type' requires option max_size
nanopb_generator.py fails
```

With this, the generated struct has:
```c
typedef struct {
    double latitude;
    double longitude;
    float  altitude_m;
    char   fix_type[8];   // exactly 8 bytes, no malloc
    uint32_t satellites;
} GpsData;
```

**The general rule:** for every `string` or `bytes` field in your .proto, you must add a `max_size` line in `.options`. For repeated fields (arrays), you need `max_count`. These are nanopb-specific annotations that don't appear in standard protobuf — they exist only to make static allocation possible.

---

### 1.10 pb_encode return value — what it means when it returns true

**The intuition that's wrong:**

Many beginners check `pb_encode()` return value and think: "if it returns `true`, the output buffer contains a complete, valid message." This is almost correct, but missing a critical detail.

`pb_encode()` returns `true` if encoding completed **without error**. But the output buffer may be SMALLER than its maximum allocated size. The number of bytes actually written is in `stream.bytes_written`.

```c
uint8_t buf[128];  // allocated 128 bytes
pb_ostream_t stream = pb_ostream_from_buffer(buf, sizeof(buf));

SensorFrame frame = SensorFrame_init_zero;
frame.seq = 42;
// (other fields left zero)

bool ok = pb_encode(&stream, SensorFrame_fields, &frame);

// ok == true, BUT:
// stream.bytes_written might be 4 (only seq field was encoded)
// buf[4..127] still has whatever was there before (uninitialized / previous transfer data)
```

If you tell the SPI DMA to send all 128 bytes, the Jetson receives 4 valid bytes and 124 bytes of garbage.

**The correct pattern:**

```c
bool ok = pb_encode(&stream, SensorFrame_fields, &frame);
if (!ok) {
    LOG_ERR("encode failed: %s", PB_GET_ERROR(&stream));
    return 0;
}
// Use stream.bytes_written — NOT sizeof(buf)
uint16_t payload_len = (uint16_t)stream.bytes_written;
// Write payload_len to frame header so receiver knows how many bytes to decode
```

---

### 1.11 The payload_length design pattern

**The invariant you need:**

Every SPI frame your STM32 sends must allow the Jetson to determine: "how many bytes of this 128-byte buffer are valid protobuf data?"

The standard solution is a fixed-format 4-byte header before the protobuf payload:

```
┌────────────────────────────────────────────────────┐
│  BYTE 0   │  BYTE 1   │  BYTE 2   │  BYTE 3        │
│  0xAB     │  0xCD     │  len_hi   │  len_lo        │
│  (magic)  │  (magic)  │           │                │
└────────────────────────────────────────────────────┘
│  BYTES 4..(4+len-1): protobuf payload              │
│  BYTES (4+len)..127: padding zeros                 │
└────────────────────────────────────────────────────┘
```

The magic bytes (0xAB 0xCD) help the Jetson detect frame alignment errors: if byte 0 is NOT 0xAB, the SPI receiver is out-of-sync. Reset and re-sync rather than decode garbage.

On the Jetson side:
```python
# After reading 128-byte SPI frame:
if buf[0] != 0xAB or buf[1] != 0xCD:
    stats.desync_count += 1
    return None  # skip this frame, wait for sync

payload_len = (buf[2] << 8) | buf[3]
if payload_len > 124:   # 128 - 4 byte header
    return None  # corrupt

payload = bytes(buf[4 : 4 + payload_len])
frame = SensorFrame()
frame.ParseFromString(payload)
```

---

### 1.12 What ZBus message drops look like — the silent 3-4 window pattern

**Why drops happen at 100Hz:**

A subscriber has a queue with depth N (e.g. depth 4). The IMU thread publishes every 10ms. If the subscriber thread is busy for longer than N×10ms (40ms for depth 4), the notification queue fills up. The next `zbus_chan_pub()` tries to queue a notification → queue is full → notification silently dropped.

**What you see in the data:**

The subscriber reads the *channel* (not the queue notification). When it finally gets a notification, it reads the current channel value — the latest one. The intermediate values are gone. But the Jetson doesn't know values were skipped. It sees:

```
t=0ms:   frame.seq=100, accel_x=1.23
t=10ms:  frame.seq=101, accel_x=1.24
t=20ms:  frame.seq=102, accel_x=1.23
t=30ms:  frame.seq=103, accel_x=1.23  ← same, benign?
t=40ms:  frame.seq=104, accel_x=1.23  ← same again
t=50ms:  frame.seq=105, accel_x=1.23  ← same again
t=60ms:  frame.seq=110, accel_x=1.31  ← seq jumped by 5
```

The seq jump from 105 to 110 tells you 5 notifications were dropped. On the Jetson side if you're not looking at `seq`, you just see 3–4 identical accel_x values — which could look like the sensor is stuck, or like the robot stopped moving.

This is exactly the sensorbar duplicate-reading pattern from incident #99185: `is_reliable=True` on all readings, but byte-identical consecutive values at 30ms intervals instead of the expected 10ms. Identical values across 3–4 consecutive windows is the diagnostic signature of a dropped notification OR a stale DMA buffer.

**Prevention:**
- Size the subscriber queue for burst depth: `ZBUS_SUBSCRIBER_DEFINE(packer_sub, 10)` for 100ms burst tolerance
- Check `seq` on the Jetson side and alert if `delta_seq > 1`
- Monitor `zbus_obs_get_chan_drop_cnt()` in your Zephyr shell

---

### 1.13 Stack usage from pb_encode — why your thread stack is too small

**What pb_encode actually does on the stack:**

`pb_encode()` is a recursive function. For each sub-message field it encounters, it recurses into a nested encode call. Each recursive call pushes its own local variables, state, and the `pb_ostream_t` onto the stack.

For a `SensorFrame` with `ImuData` and `WheelData` sub-messages:

```
Call stack during pb_encode(SensorFrame):
  pb_encode(SensorFrame)          ← ~120 bytes of locals
    pb_encode_tag_for_field()     ← ~24 bytes
    pb_encode_submessage(ImuData) ← ~80 bytes
      pb_encode(ImuData)          ← another ~120 bytes of locals
        pb_encode_tag_for_field() ← ~24 bytes (×7 for 7 fields)
        pb_encode_fixed32()       ← ~16 bytes
  ← back from ImuData
    pb_encode_submessage(WheelData) ← ~80 bytes
      pb_encode(WheelData)          ← another ~120 bytes of locals
```

A two-level nested message can consume ~500–750 bytes of stack just for pb_encode, before your own function's locals are counted.

**The minimum thread stack for a packer thread:**

| What's on the stack | Bytes |
|---|---|
| Thread function locals | 64 |
| `pb_encode()` for outer SensorFrame | 120 |
| `pb_encode()` for ImuData sub-msg | 200 |
| `pb_encode()` for WheelData sub-msg | 200 |
| `zbus_chan_read()` overhead | 64 |
| LOG_ERR() if encode fails | 400 |
| Safety margin (20%) | 210 |
| **Total** | **~1,258** |

Start at 2048 bytes. Use `k_thread_stack_space_get()` after 5 minutes of operation to verify unused space.

---

## PART 2 — Annotated Code Reference

---

### 2.1 ZBus channel + subscriber definition (.h and .c, full)

```c
// ============================================================
// sensors/channels.h — channel declarations visible to all files
// Include guard prevents double-inclusion
// ============================================================
#ifndef SENSORS_CHANNELS_H
#define SENSORS_CHANNELS_H

#include <zephyr/zbus/zbus.h>

// Message structs — only plain data (no pointers! ZBus copies by value).
// Pointers would become dangling after the publish memcpy.
struct imu_data {
    float    accel_x, accel_y, accel_z;   // m/s² after scaling
    float    gyro_x,  gyro_y,  gyro_z;    // rad/s after scaling
    int64_t  timestamp_us;                // k_ticks_to_us_near64 at read time
};

struct gps_data {
    double   latitude;
    double   longitude;
    float    altitude_m;
    uint8_t  satellites;
    bool     has_fix;                     // false = all lat/lon/alt are zero/invalid
    int64_t  timestamp_us;
};

// ZBUS_CHAN_DECLARE: declares the channel symbol as extern.
// This lets any .c file that includes this header call zbus_chan_pub/read on it.
// The actual memory lives in channels.c (defined with ZBUS_CHAN_DEFINE there).
ZBUS_CHAN_DECLARE(imu_chan);
ZBUS_CHAN_DECLARE(gps_chan);

#endif /* SENSORS_CHANNELS_H */
```

```c
// ============================================================
// sensors/channels.c — channel definitions (compiled exactly once)
// ============================================================
#include "channels.h"

// ZBUS_CHAN_DEFINE(name, msg_type, validator_fn, user_data, observers, init_val)
//
// name:         C symbol name — must match ZBUS_CHAN_DECLARE in the header
// msg_type:     the struct type this channel carries
// validator_fn: called on pub() before storing; NULL = accept all
//               signature: bool fn(const struct zbus_channel *chan);
// user_data:    arbitrary pointer attached to channel (NULL if unused)
// observers:    compile-time observer list. Use ZBUS_OBSERVERS_EMPTY if adding
//               observers at runtime via ZBUS_CHAN_ADD_OBS
// init_val:     ZBUS_MSG_INIT(0) zero-initializes the channel's internal copy
//
ZBUS_CHAN_DEFINE(
    imu_chan,
    struct imu_data,
    NULL,
    NULL,
    ZBUS_OBSERVERS_EMPTY,
    ZBUS_MSG_INIT(0)
);

ZBUS_CHAN_DEFINE(
    gps_chan,
    struct gps_data,
    NULL,
    NULL,
    ZBUS_OBSERVERS_EMPTY,
    ZBUS_MSG_INIT(0)
);
```

```c
// ============================================================
// packer/packer.c — subscriber definition and registration
// ============================================================
#include <zephyr/zbus/zbus.h>
#include "channels.h"

// ZBUS_SUBSCRIBER_DEFINE(name, queue_depth)
//
// name:        C symbol name — the observer's identity
// queue_depth: how many pending notifications can queue up before oldest is dropped.
//              At 100Hz, depth=10 tolerates up to 100ms of subscriber backlog.
//              Depth=1 means: if the subscriber misses one publish, it loses the notification.
//
ZBUS_SUBSCRIBER_DEFINE(packer_sub, 10);

// ZBUS_CHAN_ADD_OBS(channel, observer, priority)
//
// Registers packer_sub as an observer of imu_chan at observation priority 5.
// Priority controls the order callbacks fire when multiple observers exist.
// Lower priority number = earlier notification.
//
// This is a RUNTIME registration — requires CONFIG_ZBUS_RUNTIME_OBSERVERS=y in prj.conf.
// For compile-time registration on the channel itself, use ZBUS_CHAN_DEFINE with
// ZBUS_OBSERVERS(packer_sub) instead of ZBUS_OBSERVERS_EMPTY.
//
// We call this from an init function rather than using compile-time registration
// so channels.c doesn't need to know about packer.c (maintains separation of concerns).
//
int packer_init(void)
{
    int rc;

    rc = zbus_chan_add_obs(&imu_chan, &packer_sub, K_MSEC(10));
    if (rc != 0) {
        LOG_ERR("Failed to register packer_sub on imu_chan: %d", rc);
        return rc;
    }

    rc = zbus_chan_add_obs(&gps_chan, &packer_sub, K_MSEC(10));
    if (rc != 0) {
        LOG_ERR("Failed to register packer_sub on gps_chan: %d", rc);
        return rc;
    }

    return 0;
}
```

---

### 2.2 Safe publish from thread — and the WRONG pattern (publish from ISR with listener)

```c
// ============================================================
// CORRECT: publish from a thread with appropriate timeout
// ============================================================
void imu_thread_fn(void *a, void *b, void *c)
{
    struct imu_data msg;

    while (1) {
        // ... read ICM-42688 registers ...

        msg.timestamp_us = k_ticks_to_us_near64(k_uptime_ticks());
        // k_uptime_ticks() is the highest-resolution time source.
        // Convert to microseconds for the Jetson's ROS2 timestamp field.

        // K_MSEC(1): wait up to 1ms for the channel mutex.
        // If another thread holds the mutex, we wait rather than dropping data.
        // For a 100Hz sensor, 1ms wait is fine — we never miss our next sample window.
        int rc = zbus_chan_pub(&imu_chan, &msg, K_MSEC(1));

        if (rc == -EAGAIN) {
            // Mutex was held for >1ms — something is wrong.
            // A listener callback might be blocking (see WRONG pattern below).
            LOG_WRN("imu_chan pub timeout — check listeners");
        } else if (rc == -ENOMSG) {
            // Validator rejected the message (if you have a validator).
            LOG_ERR("imu_chan validator rejected message");
        }

        k_msleep(10);   // 100Hz
    }
}
```

```c
// ============================================================
// WRONG: listener that does slow work
//
// This listener is called inside zbus_chan_pub(), in imu_thread's context.
// If it takes >1ms, imu_thread's pub() call returns -EAGAIN on the next tick.
// ============================================================

// DON'T DO THIS — listener doing slow I2C or logging
static void wrong_listener_cb(const struct zbus_channel *chan)
{
    ARG_UNUSED(chan);
    struct imu_data data;
    zbus_chan_read(chan, &data, K_NO_WAIT);

    // WRONG: printk acquires a UART lock internally — can take 1-5ms
    printk("IMU: ax=%.3f ay=%.3f az=%.3f\n",
           data.accel_x, data.accel_y, data.accel_z);

    // WRONG: SPI write from a listener also takes time and may sleep
    // spi_write(...);
}
ZBUS_LISTENER_DEFINE(wrong_logger, wrong_listener_cb);

// ============================================================
// CORRECT: listener that only sets an atomic flag
// The actual logging runs in a separate thread that reads the flag.
// ============================================================
static atomic_t imu_ready_flag = ATOMIC_INIT(0);

static void correct_listener_cb(const struct zbus_channel *chan)
{
    ARG_UNUSED(chan);
    // atomic_set is lock-free and takes ~1 CPU cycle
    atomic_set(&imu_ready_flag, 1);
}
ZBUS_LISTENER_DEFINE(imu_flagger, correct_listener_cb);
```

---

### 2.3 ZBus subscribe in a thread — the blocking zbus_sub_wait() loop

```c
// ============================================================
// packer/packer_thread.c — the canonical subscriber thread pattern
// ============================================================
#include <zephyr/kernel.h>
#include <zephyr/zbus/zbus.h>
#include <zephyr/logging/log.h>
#include "channels.h"

LOG_MODULE_REGISTER(packer, LOG_LEVEL_DBG);

// packer_sub was defined in packer.c (section 2.1)
// The EXTERN declaration makes it visible here
extern const struct zbus_observer packer_sub;

static void packer_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;  // will point to whatever channel notified us
    struct imu_data imu;
    struct gps_data gps;
    bool imu_fresh = false;          // track whether we have new data to encode

    while (1) {
        // zbus_sub_wait: blocks until ANY channel we've subscribed to publishes.
        // K_FOREVER: no timeout — wait as long as needed.
        // chan: output — tells us WHICH channel triggered this wake.
        //
        // At 100Hz IMU, this wakes approximately every 10ms.
        // Multiple channels can fire between wakes — zbus_sub_wait returns
        // one notification at a time. The while(1) loop handles the rest.
        //
        int rc = zbus_sub_wait(&packer_sub, &chan, K_FOREVER);
        if (rc != 0) {
            LOG_WRN("zbus_sub_wait error: %d", rc);
            continue;
        }

        // Determine which channel fired and read its latest value.
        // We must still call zbus_chan_read — the notification only wakes us,
        // it does NOT automatically hand us the data.
        if (chan == &imu_chan) {
            // K_NO_WAIT: channel was just published and mutex should be available.
            // If we get -EAGAIN here, something is very wrong; log and continue.
            rc = zbus_chan_read(&imu_chan, &imu, K_NO_WAIT);
            if (rc == 0) {
                imu_fresh = true;
            }
        } else if (chan == &gps_chan) {
            zbus_chan_read(&gps_chan, &gps, K_NO_WAIT);
            // GPS at 1Hz — just store it; use it in the next encode cycle
        }

        // Only encode when we have a fresh IMU reading
        // (GPS might not have arrived yet on the first few ticks)
        if (imu_fresh) {
            encode_and_swap(&imu, &gps);
            imu_fresh = false;
        }
    }
}

// Stack of 2048 bytes: pb_encode needs ~500, LOG_ERR needs ~400, rest is margin
// Priority 5: lower than IMU thread (IMU should publish before we try to read)
K_THREAD_DEFINE(packer_tid, 2048, packer_thread_fn, NULL, NULL, NULL, 5, 0, 0);
```

---

### 2.4 Complete .proto file for the sensor message

```protobuf
// proto/sensor_frame.proto
//
// This file is the contract between the STM32 firmware and the Jetson ROS2 node.
// NEVER change a field number. NEVER remove a field. Add new fields at the end only.
//
syntax = "proto3";

// ImuData: accelerometer + gyroscope from ICM-42688
// Field numbers are permanent identifiers — they determine wire encoding.
// The names are for humans only; on the wire, only the numbers travel.
message ImuData {
    float  accel_x      = 1;   // m/s² — scaled from raw ADC by firmware
    float  accel_y      = 2;
    float  accel_z      = 3;
    float  gyro_x       = 4;   // rad/s
    float  gyro_y       = 5;
    float  gyro_z       = 6;
    uint64 timestamp_us = 7;   // microseconds since Zephyr boot (k_uptime_ticks)
}

// GpsData: NMEA GPGGA sentence parsed on STM32
message GpsData {
    double latitude     = 1;   // degrees, positive=N, negative=S
    double longitude    = 2;   // degrees, positive=E, negative=W
    float  altitude_m   = 3;   // meters above MSL
    uint32 satellites   = 4;   // number of satellites in fix
    bool   has_fix      = 5;   // FALSE → all position fields are zero/invalid
    string fix_type     = 6;   // "no_fix", "2D", "3D" — max_size set in .options
    uint64 timestamp_us = 7;
}

// OdomData: wheel encoder odometry
message OdomData {
    float  speed_fl     = 1;   // m/s, front-left wheel
    float  speed_fr     = 2;
    float  speed_rl     = 3;
    float  speed_rr     = 4;
    float  heading_rad  = 5;   // integrated yaw from encoders
    uint64 timestamp_us = 6;
}

// SensorFrame: top-level message sent every 10ms over SPI
message SensorFrame {
    uint32  seq            = 1;   // sequence counter, wraps at 2^32. Jetson uses this
                                  // to detect drops (delta_seq > 1 = missing frames).
    ImuData   imu          = 2;   // always present
    GpsData   gps          = 3;   // present if GPS module connected
    OdomData  odom         = 4;   // present if wheel encoders available
    uint32    fw_version   = 5;   // firmware build version for compatibility check
    // Field 6 reserved for future use — do not add here until needed
}
```

```ini
# proto/sensor_frame.options
# nanopb-specific size constraints.
# Without these, nanopb_generator fails on string fields with:
#   "Field GpsData.fix_type requires option max_size"
#
# max_size includes the null terminator.
# "no_fix" = 6 chars + 1 null = 7 → round up to 8.
#
GpsData.fix_type    max_size:8
```

---

### 2.5 nanopb CMake integration — the code generation custom command

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sensor_packer)

# --- nanopb code generation ---
#
# nanopb ships with a CMake helper (nanopb.cmake) that adds:
#   nanopb_generate_cpp(SRCS HDRS <proto_files...>)
#
# Under the hood, this creates a custom_command that runs:
#   python -m nanopb_generator --output-dir=<build_dir> <proto_file>
# producing sensor_frame.pb.c and sensor_frame.pb.h in the build directory.
#
# If nanopb is not in path, set NANOPB_SRC_ROOT_FOLDER before find_package.
#

# Tell nanopb where to find the .options file (same dir as .proto)
set(NANOPB_OPTIONS_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto)

# Generate .pb.c and .pb.h from our .proto
nanopb_generate_cpp(PROTO_SRCS PROTO_HDRS
    proto/sensor_frame.proto
)

# App sources
target_sources(app PRIVATE
    src/main.c
    src/imu_thread.c
    src/packer_thread.c
    src/channels.c
    ${PROTO_SRCS}            # the generated sensor_frame.pb.c
)

# Generated headers land in the build directory — add it to include path
# so #include "sensor_frame.pb.h" works without specifying a path
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    src/
    proto/
)
```

```ini
# prj.conf additions for ZBus + nanopb
CONFIG_ZBUS=y
CONFIG_ZBUS_RUNTIME_OBSERVERS=y    # allow ZBUS_CHAN_ADD_OBS at runtime
CONFIG_ZBUS_CHANNEL_NAME=y         # include channel names in observer debug output
CONFIG_NANOPB=y                    # link nanopb encode/decode library
```

---

### 2.6 pb_encode() with pb_ostream_from_buffer() — annotated with PB_GET_ERROR

```c
// ============================================================
// packer/encoder.c — the encode() function used by packer_thread
// Returns number of bytes written, or 0 on error.
// ============================================================
#include <pb_encode.h>
#include "sensor_frame.pb.h"
#include "channels.h"

// Maximum frame size: 4-byte header + protobuf payload
// Calculated: SensorFrame max ~128 bytes protobuf + 4 header = 132.
// Round up to 160 for safety.
#define FRAME_HEADER_SIZE  4
#define PROTO_BUF_MAX     160
#define FRAME_TOTAL_SIZE  (FRAME_HEADER_SIZE + PROTO_BUF_MAX)

// encode_sensor_frame: fills out_frame and returns total frame bytes written.
// out_frame must be at least FRAME_TOTAL_SIZE bytes.
uint16_t encode_sensor_frame(
    const struct imu_data   *imu,
    const struct gps_data   *gps,
    uint8_t                 *out_frame,
    uint16_t                 out_frame_size)
{
    if (out_frame_size < FRAME_TOTAL_SIZE) {
        return 0;  // caller's buffer too small
    }

    // ---- 1. Build the protobuf data structure ----
    // SensorFrame_init_zero is generated by nanopb and sets ALL fields to zero
    // AND all has_* flags to false. ALWAYS start with this.
    // If you use an uninitialized struct, leftover stack bytes look like valid fields.
    SensorFrame frame = SensorFrame_init_zero;

    // Monotonically increasing sequence number.
    // Static so it persists across calls.
    static uint32_t seq = 0;
    frame.seq = seq++;

    // IMU sub-message: always present
    frame.has_imu = true;               // set the has_ flag — without this, ImuData is skipped
    frame.imu.accel_x      = imu->accel_x;
    frame.imu.accel_y      = imu->accel_y;
    frame.imu.accel_z      = imu->accel_z;
    frame.imu.gyro_x       = imu->gyro_x;
    frame.imu.gyro_y       = imu->gyro_y;
    frame.imu.gyro_z       = imu->gyro_z;
    frame.imu.timestamp_us = imu->timestamp_us;

    // GPS sub-message: only include if we have a fix
    // If gps->has_fix is false, we still include the sub-message
    // but populate has_fix=false so the Jetson knows to ignore lat/lon/alt.
    if (gps != NULL) {
        frame.has_gps = true;
        frame.gps.latitude     = gps->latitude;
        frame.gps.longitude    = gps->longitude;
        frame.gps.altitude_m   = gps->altitude_m;
        frame.gps.satellites   = gps->satellites;
        frame.gps.has_fix      = gps->has_fix;
        strncpy(frame.gps.fix_type, gps->has_fix ? "3D" : "no_fix",
                sizeof(frame.gps.fix_type) - 1);
        frame.gps.timestamp_us = gps->timestamp_us;
    }

    // ---- 2. Set up the output stream ----
    // pb_ostream_from_buffer: wraps out_frame+HEADER in a stream.
    // Writing starts at byte FRAME_HEADER_SIZE (offset 4) to leave room for header.
    // The stream tracks bytes_written.
    uint8_t *proto_start = out_frame + FRAME_HEADER_SIZE;
    pb_ostream_t stream = pb_ostream_from_buffer(proto_start, PROTO_BUF_MAX);

    // ---- 3. Encode ----
    // pb_encode writes TLV bytes into stream.
    // Returns false if: buffer too small, required field missing, callback error.
    bool ok = pb_encode(&stream, SensorFrame_fields, &frame);

    if (!ok) {
        // PB_GET_ERROR reads the error string stored in the stream.
        // Without this, you only know "encode failed" but not why.
        LOG_ERR("pb_encode failed: %s", PB_GET_ERROR(&stream));
        return 0;
    }

    // ---- 4. Write the frame header ----
    // Magic bytes let the receiver detect frame boundary alignment.
    out_frame[0] = 0xAB;
    out_frame[1] = 0xCD;
    // payload length (big-endian): only the valid protobuf bytes
    // stream.bytes_written is the ACTUAL bytes written, not sizeof(buffer)
    uint16_t payload_len = (uint16_t)stream.bytes_written;
    out_frame[2] = (uint8_t)(payload_len >> 8);
    out_frame[3] = (uint8_t)(payload_len & 0xFF);

    // ---- 5. Zero-pad the rest of the frame ----
    // Prevents stale bytes from a previous (larger) frame from leaking.
    memset(proto_start + payload_len, 0, PROTO_BUF_MAX - payload_len);

    return FRAME_HEADER_SIZE + payload_len;
}
```

---

### 2.7 pb_decode() round-trip test on host Linux

Before porting to STM32, verify your encode/decode is correct on a Linux PC where you can print everything:

```c
// test/round_trip.c — compile and run on Linux, not Zephyr
// gcc test/round_trip.c sensor_frame.pb.c -lprotobuf-nanopb -Inanopb -o round_trip
#include <stdio.h>
#include <assert.h>
#include <pb_encode.h>
#include <pb_decode.h>
#include "sensor_frame.pb.h"

int main(void)
{
    uint8_t buf[256];

    // ---- ENCODE ----
    SensorFrame tx = SensorFrame_init_zero;
    tx.seq = 42;
    tx.has_imu = true;
    tx.imu.accel_x      = 1.23f;
    tx.imu.accel_y      = -0.45f;
    tx.imu.accel_z      = 9.81f;
    tx.imu.timestamp_us = 100000ULL;
    tx.has_gps = true;
    tx.gps.has_fix      = true;
    tx.gps.latitude     = 35.6762;
    tx.gps.longitude    = 139.6503;
    strncpy(tx.gps.fix_type, "3D", sizeof(tx.gps.fix_type));

    pb_ostream_t ostream = pb_ostream_from_buffer(buf, sizeof(buf));
    bool ok = pb_encode(&ostream, SensorFrame_fields, &tx);
    assert(ok && "encode failed");

    printf("Encoded %zu bytes\n", ostream.bytes_written);

    // Print hex dump — compare against what you expect
    for (size_t i = 0; i < ostream.bytes_written; i++) {
        printf("%02X ", buf[i]);
        if ((i+1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // ---- DECODE ----
    SensorFrame rx = SensorFrame_init_zero;

    // pb_istream_from_buffer: wraps buf for decoding.
    // CRITICAL: pass ostream.bytes_written, NOT sizeof(buf).
    // Passing sizeof(buf) makes the decoder read past the valid payload
    // and interpret zero-padding as fields — exactly the SPI bug.
    pb_istream_t istream = pb_istream_from_buffer(buf, ostream.bytes_written);
    ok = pb_decode(&istream, SensorFrame_fields, &rx);

    if (!ok) {
        // PB_GET_ERROR gives the reason — much better than a raw bool
        fprintf(stderr, "decode failed: %s\n", PB_GET_ERROR(&istream));
        return 1;
    }

    // ---- VERIFY round-trip ----
    assert(rx.seq == 42);
    assert(rx.has_imu);
    // Float comparison with small epsilon (floating point isn't exact)
    assert(fabsf(rx.imu.accel_x - 1.23f) < 0.001f);
    assert(rx.has_gps && rx.gps.has_fix);
    assert(strcmp(rx.gps.fix_type, "3D") == 0);

    printf("Round-trip test PASSED\n");
    return 0;
}
```

Build and run this on your development machine BEFORE bringing nanopb onto the STM32. If it fails here, the .proto/.options/CMakeLists.txt have a bug. If it passes here but fails on STM32, the problem is firmware-specific (stack, buffer size, DMA alignment).

---

### 2.8 The packer thread — complete annotated example

```c
// ============================================================
// packer/packer_thread.c — integrates ZBus + nanopb + double-buffer
// ============================================================
#include <zephyr/kernel.h>
#include <zephyr/zbus/zbus.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include "channels.h"
#include "encoder.h"        // encode_sensor_frame()
#include "spi_slave.h"      // spi_swap_buffer()

LOG_MODULE_REGISTER(packer, LOG_LEVEL_INF);

// Double-buffer: while the SPI DMA is transmitting buffer A,
// the packer thread encodes into buffer B.
// On the next swap, A becomes the write target and B starts transmitting.
// This ensures the SPI DMA never reads from a buffer we're writing to.
#define FRAME_SIZE  164   // 4 header + 160 max protobuf
static uint8_t frame_buf[2][FRAME_SIZE];
static uint8_t active_buf = 0;   // which buffer SPI DMA is currently sending

extern const struct zbus_observer packer_sub;

static void packer_thread_fn(void *a, void *b, void *c)
{
    const struct zbus_channel *chan;
    struct imu_data  imu  = {0};
    struct gps_data  gps  = {0};

    uint32_t imu_seq_last = 0;   // detect ZBus drops via IMU seq
    uint32_t encode_count = 0;

    LOG_INF("Packer thread started");

    while (1) {
        // Block until imu_chan or gps_chan publishes
        int rc = zbus_sub_wait(&packer_sub, &chan, K_FOREVER);
        if (rc != 0) {
            LOG_WRN("sub_wait error: %d", rc);
            continue;
        }

        if (chan == &imu_chan) {
            struct imu_data new_imu;
            if (zbus_chan_read(&imu_chan, &new_imu, K_NO_WAIT) == 0) {
                // Check for ZBus drops: if we see a seq jump > 1 here,
                // it means the packer ran slower than the IMU published.
                // (requires imu_data to include a seq field)
                imu = new_imu;
            }
        } else if (chan == &gps_chan) {
            // GPS at 1Hz — just cache it, don't encode mid-SPI-frame
            zbus_chan_read(&gps_chan, &gps, K_NO_WAIT);
            continue;   // don't trigger an encode for GPS-only updates
        }

        // Select the INACTIVE buffer to write into
        // (active_buf is currently being sent by SPI DMA — don't touch it)
        uint8_t write_buf = 1 - active_buf;

        uint16_t frame_len = encode_sensor_frame(
            &imu, &gps,
            frame_buf[write_buf],
            sizeof(frame_buf[write_buf])
        );

        if (frame_len == 0) {
            LOG_ERR("encode_sensor_frame failed — skipping swap");
            continue;
        }

        // Atomic buffer swap: tell SPI slave to start using write_buf for next transfer.
        // spi_swap_buffer() waits for the current DMA transfer to complete,
        // then switches the DMA source pointer to the new buffer.
        spi_swap_buffer(frame_buf[write_buf], FRAME_SIZE);
        active_buf = write_buf;

        encode_count++;
        if (encode_count % 1000 == 0) {
            LOG_INF("Packer: %u frames encoded", encode_count);
        }
    }
}

K_THREAD_DEFINE(
    packer_tid,
    2048,               // 2048: pb_encode stack + LOG overhead + safety margin
    packer_thread_fn,
    NULL, NULL, NULL,
    5,                  // priority 5: below IMU thread (typically at 3), above idle
    0,
    0
);
```

---

## PART 3 — Gotcha Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| Jetson sees identical sensor values for 3–4 consecutive 10ms frames | ZBus subscriber queue full — packer thread is too slow; notification dropped; channel holds same stale value until packer finally wakes | Check `zbus_obs_get_chan_drop_cnt(&packer_sub)` in Zephyr shell. On Jetson, check `delta_seq` — a jump from N to N+4 confirms 3 drops | Increase subscriber queue depth: `ZBUS_SUBSCRIBER_DEFINE(packer_sub, 10)`. Reduce packer thread work, or increase its priority |
| `pb_encode()` returns `false` with no visible error message | Missing `max_size` annotation in `.options` file for a string field | The PB_GET_ERROR string will say "Field allocation failed" or "bytes_written exceeded" | Add `FieldName.field_name  max_size:N` to `sensor_frame.options`. Regenerate .pb.h/.pb.c |
| Jetson successfully decodes some frames but gets garbage floats when GPS loses fix | Variable-length protobuf output + fixed-size SPI DMA: when GPS fields go to zero they're omitted, producing fewer bytes; Jetson reads stale bytes from unflushed DMA buffer | On Jetson: print `len(raw_bytes)` before decode — it will be fixed (128) but payload valid only for first N bytes | Add explicit `payload_length` 2-byte header. Zero-pad the remainder in `encode_sensor_frame()`. On Jetson, decode only `payload_length` bytes |
| `pb_encode()` returns `true` but decoded fields are zero on the Jetson | `pb_istream_from_buffer()` called with `sizeof(buf)` instead of `stream.bytes_written` — decoder reads zero-padded region and interprets zeros as "field not present" | In Jetson Python: `print(frame.HasField('imu'))` prints False even though IMU was encoded | Pass `ostream.bytes_written` (C) or `payload_length` from header (Python) to the decode call, not the buffer size |
| Hard fault or stack overflow in packer thread | pb_encode nested sub-messages use ~500–750 bytes of stack; combined with LOG_ERR (~400 bytes) exceeds 1024-byte default stack | `CONFIG_STACK_SENTINEL=y` catches it with "Stack sentinel violation". Without it: random corruption with no error | Set packer thread stack to 2048 minimum. Check high-water mark: `k_thread_stack_space_get(&packer_tid, &unused)` — target <80% used |
| `zbus_chan_pub()` returns `-EAGAIN` consistently | A listener callback is blocking — it holds the channel mutex while doing slow I/O | Add timing: `uint32_t t0 = k_uptime_get_32(); pub(); LOG_DBG("pub took %u ms", k_uptime_get_32()-t0)` — if >1ms, a listener is the culprit | Move slow work out of listeners into subscribers. Listeners must complete in <50µs |
| Wrong priority on packer thread — ZBus drops still high even with depth=10 | Packer runs at lower priority than other threads which starve it; it processes its queue faster than it reads ZBus | Use `west build -t menuconfig` → Task-Aware Debugger, or enable `CONFIG_THREAD_MONITOR=y` and check via shell `kernel threads` | Set packer priority numerically close to (but below) the IMU reading thread. If IMU is priority 3, make packer priority 4 |
| First 1–4 bytes of EVERY SPI frame are 0x00 on Jetson | SPI slave DMA pre-arming race: DMA armed in CS interrupt callback, but CS goes low and clock starts before DMA is ready | Logic analyzer: trigger on CS falling edge, observe first byte(s) on MISO line — they will be 0x00 while subsequent bytes are correct | Arm SPI slave DMA in the *previous transfer complete* callback, not on CS assert. See docs/runbooks/spi-slave-race.md |
| `ZBUS_CHAN_DECLARE` in .h file causes linker error "multiple definition of imu_chan" | ZBUS_CHAN_DEFINE used in a header file instead of a .c file — it allocates memory, so it appears in every translation unit that includes the header | `grep -r ZBUS_CHAN_DEFINE` — should appear only in .c files | Move `ZBUS_CHAN_DEFINE` to a .c file. Use `ZBUS_CHAN_DECLARE` in the shared header |
| ZBus compile error: `CONFIG_ZBUS_RUNTIME_OBSERVERS not set` | `zbus_chan_add_obs()` used without enabling runtime observers | Build error mentions `CONFIG_ZBUS_RUNTIME_OBSERVERS` | Add `CONFIG_ZBUS_RUNTIME_OBSERVERS=y` to prj.conf; or register observers compile-time with `ZBUS_OBSERVERS(sub_name)` in `ZBUS_CHAN_DEFINE` |
| nanopb decode produces struct with zeroed sub-message fields even though sender set them | `has_imu` flag not set before encoding — proto3 sub-messages are only included if `has_*` is `true` | Hex-dump the encoded bytes on STM32: a SensorFrame without ImuData will not contain field-2 (tag 0x12) | Set `frame.has_imu = true` before filling `frame.imu.*` fields |

---

## Quick Reference Card

```
ZBus prj.conf:
  CONFIG_ZBUS=y
  CONFIG_ZBUS_RUNTIME_OBSERVERS=y   (for ZBUS_CHAN_ADD_OBS at runtime)
  CONFIG_ZBUS_CHANNEL_NAME=y        (debug: channel names in output)

nanopb prj.conf:
  CONFIG_NANOPB=y

ZBus API summary:
  ZBUS_CHAN_DEFINE(name, type, validator, user_data, observers, init)  ← in .c
  ZBUS_CHAN_DECLARE(name)                                               ← in .h
  ZBUS_SUBSCRIBER_DEFINE(name, queue_depth)                            ← in .c
  ZBUS_LISTENER_DEFINE(name, callback_fn)                              ← in .c
  zbus_chan_pub(&chan, &msg, timeout)    ← write new value
  zbus_chan_read(&chan, &msg, timeout)   ← read latest value
  zbus_sub_wait(&sub, &chan_ptr, timeout) ← block until notification
  zbus_chan_add_obs(&chan, &obs, timeout) ← runtime subscription

nanopb API summary:
  SensorFrame frame = SensorFrame_init_zero;  ← ALWAYS start with this
  pb_ostream_t out = pb_ostream_from_buffer(buf, size);
  bool ok = pb_encode(&out, SensorFrame_fields, &frame);
  uint16_t written = (uint16_t)out.bytes_written;  ← NOT sizeof(buf)!
  if (!ok) { LOG_ERR("%s", PB_GET_ERROR(&out)); }

  pb_istream_t in = pb_istream_from_buffer(buf, written);  ← pass written, not sizeof
  bool ok = pb_decode(&in, SensorFrame_fields, &frame);
  if (!ok) { LOG_ERR("%s", PB_GET_ERROR(&in)); }

SPI frame layout:
  [0]     = 0xAB  (magic)
  [1]     = 0xCD  (magic)
  [2]     = payload_len >> 8
  [3]     = payload_len & 0xFF
  [4..N]  = protobuf bytes (exactly payload_len bytes)
  [N+1..] = zero padding

Stack rules:
  Packer thread calling pb_encode (nested 2 sub-messages): min 2048 bytes
  ZBus subscriber thread without LOG: min 1024 bytes
  Listener callback: must complete in <50µs, no sleeping

ZBus drop detection:
  On STM32: zbus_obs_get_chan_drop_cnt(&observer)
  On Jetson: delta_seq = frame.seq - prev_seq; if delta_seq > 1 → drop
  Symptom:   identical sensor values across 3–4 consecutive 10ms windows

Proto3 zero-omission rule:
  Fields equal to their default value (0, false, "") are NOT encoded.
  Always use payload_length header — never decode sizeof(buffer) bytes.
  Always test GPS-loses-fix edge case before deploying.
```
