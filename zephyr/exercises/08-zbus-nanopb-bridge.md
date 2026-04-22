# Exercises: Multi-Sensor ZBus + nanopb Bridge
### Covers: Deep-dive session 09 — SensorFrame proto, ZBus multi-channel, length-prefix framing, 3 failure points

---

## Section A — Conceptual Questions

**A1.** Your `SensorFrame` protobuf has three sub-messages: `ImuData`, `EncoderData`, `GpsData`.
During encoding with `pb_encode()`, which fields are **omitted** from the output bytes?
Give one example for each sub-message where a real sensor reading would be silently omitted.

<details><summary>Answer</summary>

Proto3 omits any field equal to its type's default: `0` for numbers, `false` for booleans,
`""` for strings, and absent for optional sub-messages with zero-value fields.

**ImuData:** `angular_velocity_z = 0.0` — robot is stationary, not rotating. Field omitted. The
Jetson decoder sees 0.0 (default), which is correct but indistinguishable from "encoder didn't
encode this". If the IMU driver had a bug returning 0 for z-axis, you'd never know from the wire.

**EncoderData:** `left_vel = 0` and `right_vel = 0` — robot is stopped. Both int16 fields omitted.
`timestamp_ms = 0` would also be omitted (though timestamp should never be 0 in practice — if it
is, check that the encoder timer is initialised before the first CAN frame is sent).

**GpsData:** `latitude = 0.0`, `longitude = 0.0` — robot is in BUS-OFF or GPS has no fix while
exactly on the equator/prime meridian (unlikely but theoretically valid). `has_fix = false` is
also omitted (bool default). The entire GpsData sub-message encodes to 0 bytes. The Jetson cannot
tell "no GPS" from "GPS locked at 0°, 0°, 0m".

**Rule:** Always include an explicit `has_fix` / `is_valid` field and ensure it is non-default
(i.e., `true`) for any sub-message you need to distinguish from "absent".

</details>

---

**A2.** Your SPI DMA frame is 130 bytes: 2-byte length prefix + up to 128 bytes of protobuf payload.
Explain what happens when `pb_encode()` produces 60 bytes for one SensorFrame but 110 bytes for
the next (GPS fix just acquired). How does the receiver know which is which?

<details><summary>Answer</summary>

The 2-byte big-endian length prefix tells the receiver exactly how many protobuf bytes to decode:

```
Frame N:   [0x00][0x3C] [60 bytes of proto] [68 bytes of stale DMA buffer data]
Frame N+1: [0x00][0x6E] [110 bytes of proto][18 bytes of stale DMA buffer data]
```

The Jetson reads `len = (buf[0] << 8) | buf[1]` and then decodes only those bytes:
```python
proto_bytes = frame[2 : 2 + length]
msg = SensorFrame()
msg.ParseFromString(proto_bytes)
```

The trailing stale bytes after `2 + length` are ignored. This is why the DMA buffer must be
**cleared to zero before each fill** (`memset(dma_buf, 0, 130)`), otherwise the stale length
prefix bytes from the previous frame could corrupt the next frame if `pb_encode()` produces fewer
than 2 prefix bytes (impossible in this design, but defensive).

**If you skip the length prefix** and always call `ParseFromString` on all 128 bytes: proto3
decodes the first valid protobuf fields, then hits the stale bytes from the previous frame.
Because proto3 is forward-compatible, it will attempt to parse the stale bytes as unknown fields
— silently producing a partially-corrupt message with no error.

</details>

---

**A3.** You have three ZBus channels: `imu_chan`, `encoder_chan`, `gps_chan`. The packer thread
subscribes to all three and packs them into one SensorFrame every 10ms. Explain the race condition
that can produce a SensorFrame where `imu` data is 9ms old but `encoder` data is 0.1ms old.

<details><summary>Answer</summary>

ZBus channels store only the **latest value** — there is no queue for values. The packer thread
wakes on `k_msem_take` (or `zbus_sub_wait`) when any channel is published. If:

1. IMU publishes at t=0ms.
2. Packer wakes and calls `zbus_chan_read(&imu_chan)` at t=0ms → 0ms old. ✓
3. Packer sleeps waiting for next wakeup.
4. Encoder publishes at t=9.9ms.
5. IMU publishes at t=9.95ms.
6. Encoder publishes at t=9.99ms.
7. Packer wakes (triggered by encoder at t=9.99ms).
8. Packer reads `imu_chan` → gets the t=9.95ms value (4ms old, not 9ms — close enough).

The scenario for 9ms-old IMU: the packer wakes on encoder publish at t=9ms. IMU has not yet
published in this cycle (IMU thread was preempted). Packer reads the IMU channel and gets the
t=0ms value (9ms stale). Then IMU publishes at t=10ms — too late.

**Mitigation:** Sample all channels immediately before the DMA buffer fill (not on wakeup), using
`K_NO_WAIT`. Accept that all readings are "as fresh as the last publish in the 10ms window".
Alternatively, use a dedicated 10ms periodic timer to trigger the pack, rather than waking on
channel events. Timestamps inside each sub-message reflect the actual measurement time, so the
Jetson can detect staleness regardless.

</details>

---

**A4.** `pb_encode()` returns `false`. Your code logs an error and re-sends the previous frame.
Name three distinct root causes that make `pb_encode()` return false, and the fix for each.

<details><summary>Answer</summary>

1. **Output buffer too small.** The `pb_ostream_t` is sized at 128 bytes but the encoded message
   exceeds that. `pb_encode` writes until full then returns false.
   Fix: increase buffer to the proto's `SensorFrame_size` compile-time constant (generated by
   nanopb from the .proto), or use `pb_get_encoded_size()` first.

2. **Required field not initialised (proto2 behaviour — less common in proto3).** In nanopb,
   if a field is marked `required` in the .proto and the struct field is not set, `pb_encode`
   returns false. Proto3 has no required fields, but nanopb .proto files can mix versions.
   Fix: audit the .proto — use proto3 syntax, all fields are optional.

3. **Custom encode callback returns false.** If you used a callback field (e.g. `bytes` field
   with a custom encoder for the GPS NMEA raw string) and that callback returns false (e.g.
   string is longer than allowed), the whole encode fails.
   Fix: validate the data before passing to the callback, ensure the callback's output stream
   has enough capacity.

</details>

---

## Section B — Practical / Debug Scenarios

**B1.** Your Jetson receives SensorFrame messages and logs `ParseFromString failed: invalid tag` on
approximately every 50th frame. Other frames decode correctly. The STM32 `pb_encode()` always
returns true. What is the most likely cause?

<details><summary>Answer</summary>

**DMA cache coherency or SPI frame boundary corruption.** On approximately every 50th transfer,
the Cortex-M7 D-cache has not been flushed for the DMA output buffer. The Jetson receives
either:
- Stale bytes from a previous frame (cache not flushed before DMA start), or
- Partially-written bytes (cache line partially updated).

The `invalid tag` error means the Jetson's protobuf decoder hit a byte that doesn't correspond
to any valid field tag in `SensorFrame`. Proto3 field tags have the format `(field_number << 3) |
wire_type`. A stale byte from a previous frame's payload in the wrong position produces an
unrecognised tag.

**Diagnosis:** Log the raw bytes of a failing frame on the Jetson. Check if bytes after position
~60 match the payload of an earlier frame.

**Fix:** Ensure `SCB_CleanDCache_by_Addr(dma_buf, sizeof(dma_buf))` is called after writing the
proto payload and **before** starting the DMA transfer. Place `dma_buf` in a cache-coherent
region (`__nocache`) as an alternative.

</details>

---

**B2.** Your `sensor_frame.proto` defines GPS lat/lon as `float` (32-bit). You notice that at
53°N, 6°W your position jumps by ±10m randomly. You switch to `double` (64-bit) and it stops.
Explain why.

<details><summary>Answer</summary>

A `float32` has 24 bits of mantissa, giving ~7 significant decimal digits. At latitude 53°N:
- 53.36134° in float32 = `0x42558A71` ≈ 53.36134338...

The spacing between adjacent float32 values at 53° is:
- One ULP at 53 ≈ 53 × 2^(−23) × 2^6 ≈ 53 × 7.6e-6 ≈ **4.0e-4 degrees ≈ 44 metres**

So a `float32` lat/lon can only represent positions spaced ~44m apart at 53°N. Any sub-44m
position is rounded to the nearest representable float, causing the apparent 10m jumps.

`double` (64-bit, 53-bit mantissa) has ULP ≈ 53 × 1.4e-14 ≈ **1.5e-12 degrees ≈ 0.17 μm** —
sub-millimetre precision.

**Rule:** Always use `double` for GPS coordinates. `float` is sufficient for velocities,
accelerations, and small-range values. Check your `sensor_frame.proto` — nanopb generates `float`
for `.proto` `float` type, `double` for `.proto` `double` type.

</details>

---

## Section C — Code Reading

**C1.** Find all bugs in this packer thread:

```c
void packer_thread(void *a, void *b, void *c) {
    struct sensor_frame_t frame;
    uint8_t dma_buf[130];
    SensorFrame proto_msg = SensorFrame_init_zero;

    while (1) {
        zbus_sub_wait(&packer_sub, &frame.imu_chan, K_FOREVER);

        /* Read all channels */
        zbus_chan_read(&imu_chan,     &frame.imu,     K_NO_WAIT);
        zbus_chan_read(&encoder_chan, &frame.encoder, K_NO_WAIT);
        zbus_chan_read(&gps_chan,     &frame.gps,     K_NO_WAIT);

        /* Encode */
        pb_ostream_t stream = pb_ostream_from_buffer(dma_buf + 2, 128);
        fill_proto(&proto_msg, &frame);
        pb_encode(&stream, SensorFrame_fields, &proto_msg);

        /* Length prefix */
        dma_buf[0] = (stream.bytes_written >> 8) & 0xFF;
        dma_buf[1] = stream.bytes_written & 0xFF;

        /* Trigger SPI DMA */
        spi_write(spi_dev, &spi_cfg, &spi_buf_set);
    }
}
```

<details><summary>Answer</summary>

**Four bugs:**

1. **`zbus_sub_wait` first argument is wrong.** `zbus_sub_wait` takes a `const struct
   zbus_observer *` (the subscriber) and a `const struct zbus_channel **` (output, set to the
   channel that triggered). The second argument `&frame.imu_chan` is a field of `sensor_frame_t`,
   not a channel pointer. This should be:
   ```c
   const struct zbus_channel *chan;
   zbus_sub_wait(&packer_sub, &chan, K_FOREVER);
   ```

2. **`pb_encode()` return value ignored.** If encoding fails (buffer too small, callback error),
   `stream.bytes_written` may be partially filled. The length prefix will be wrong, causing the
   Jetson to try to parse a truncated or zero-length message. Add:
   ```c
   if (!pb_encode(&stream, SensorFrame_fields, &proto_msg)) {
       LOG_ERR("pb_encode failed: %s", PB_GET_ERROR(&stream));
       continue;
   }
   ```

3. **`proto_msg` is not reset between iterations.** `SensorFrame_init_zero` only runs once at
   declaration. If GPS has a fix in frame N but loses it in frame N+1, the GPS fields from frame
   N remain in `proto_msg` (proto3 does not zero them on partial encode). Add:
   ```c
   proto_msg = (SensorFrame)SensorFrame_init_zero;
   ```
   at the top of each loop iteration.

4. **No D-cache flush before `spi_write`.** `dma_buf` is in SRAM. The CPU writes proto bytes and
   the length prefix via cache lines. Without `SCB_CleanDCache_by_Addr(dma_buf, 130)`, the DMA
   controller reads stale bytes from main SRAM that the cache has not yet written back.
   Insert the flush between the length prefix write and `spi_write`.

</details>

---
