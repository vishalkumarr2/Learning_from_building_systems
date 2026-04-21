# nanopb — Protocol Buffers on Bare Metal

## Why Not Just Send Raw C Structs?

```c
// Raw struct over SPI — looks simple, breaks in practice:
struct imu_data {
    uint8_t  status;   // 1 byte
    // [3 bytes padding inserted by compiler!]
    float    accel_x;  // 4 bytes
    float    accel_y;  // 4 bytes
};

// Problems:
// 1. Padding:      compiler adds invisible bytes between fields
// 2. Endianness:   STM32=LE, some hosts=BE — floats look wrong
// 3. Versioning:   add a field → breaks ALL existing receivers
// 4. No schema:    receiver must know exact layout via out-of-band agreement
```

**nanopb** solves all four. It's Google Protocol Buffers compiled for microcontrollers:
- No heap (no malloc)
- No dynamic allocation
- Deterministic encode/decode time
- Typically 5-15% smaller than raw struct (varint compression for small integers)

---

## The .proto File — The Shared Contract

```protobuf
// sensor_frame.proto
// This file is shared between STM32 firmware and Jetson software
syntax = "proto3";

message ImuData {
    float  accel_x      = 1;   // field number 1
    float  accel_y      = 2;
    float  accel_z      = 3;
    float  gyro_x       = 4;
    float  gyro_y       = 5;
    float  gyro_z       = 6;
    uint64 timestamp_us = 7;
}

message WheelData {
    float  speed_fl     = 1;
    float  speed_fr     = 2;
    float  speed_rl     = 3;
    float  speed_rr     = 4;
    uint64 timestamp_us = 5;
}

message SensorFrame {
    uint32    seq   = 1;
    ImuData   imu   = 2;
    WheelData wheel = 3;
}
```

**Field numbers** (1, 2, 3...) are what get encoded on the wire — NOT the names.
This is what makes versioning safe:
- Field 1 always means `accel_x` — forever
- Add field 8 for temperature → old receivers just skip it (unknown fields ignored)
- Never reuse a retired field number

---

## Wire Format — What Bytes Actually Travel

Protobuf uses **TLV encoding** (Tag-Length-Value / Tag-Value):

```
Encoding ImuData { accel_x = 1.5, timestamp_us = 1000000 }

Field 1 (accel_x = 1.5f):
  Byte 0: 0x0D  = tag: (field_number=1 << 3) | wire_type=5 (32-bit fixed float)
  Bytes 1-4: 3F C0 00 00  (IEEE 754 little-endian for 1.5)
  → 5 bytes total

Field 7 (timestamp_us = 1000000):
  Byte 0: 0x38  = tag: (field_number=7 << 3) | wire_type=0 (varint)
  Bytes 1-3: C0 84 3D  (varint encoding of 1000000)
  → 4 bytes total (NOT 8 bytes like a uint64_t — varint saves space)
```

**Varint encoding** — small numbers use fewer bytes:
```
Value 0-127    → 1 byte
Value 128-16383 → 2 bytes
Value 16384-2097151 → 3 bytes
...
```
Timestamps, counters, small integers stay compact.
Floats always take exactly 5 bytes (tag + 4).

**Missing fields**: fields with default value (0 for numbers, "" for strings) are NOT encoded at all.
```
SensorFrame with seq=5, imu filled, wheel NOT filled:
→ only encodes seq + imu fields → smaller packet
```

---

## Code Generation

```bash
# Install nanopb
pip install nanopb

# Generate C files from .proto
python -m nanopb_generator sensor_frame.proto

# Produces:
#   sensor_frame.pb.h    — struct defs + field descriptor refs
#   sensor_frame.pb.c    — field tables for encoder/decoder
```

Generated `sensor_frame.pb.h`:
```c
// AUTO-GENERATED — DO NOT EDIT
typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    uint64_t timestamp_us;
} ImuData;

typedef struct {
    uint32_t seq;
    bool has_imu;        // optional field present flag
    ImuData imu;
    bool has_wheel;
    WheelData wheel;
} SensorFrame;

// Field descriptor — tells encoder which offsets map to which field numbers
extern const pb_msgdesc_t SensorFrame_msg;
#define SensorFrame_fields &SensorFrame_msg
```

---

## Encoding (STM32 / Zephyr side)

```c
#include <pb_encode.h>
#include "sensor_frame.pb.h"

// Returns number of bytes written, or 0 on error
uint16_t encode_sensor_frame(
    const struct imu_data *imu,
    const struct wheel_data *wheel,
    uint8_t *out_buf,
    size_t out_buf_size)
{
    // Create stream — writes into our static buffer
    pb_ostream_t stream = pb_ostream_from_buffer(out_buf, out_buf_size);

    // Fill the generated struct — ALWAYS start with _init_zero
    // _init_zero zeros all fields + sets has_* = false
    SensorFrame frame = SensorFrame_init_zero;

    frame.seq = g_seq++;  // monotonically increasing sequence number

    // Fill IMU sub-message
    frame.has_imu = true;
    frame.imu.accel_x      = imu->accel_x;
    frame.imu.accel_y      = imu->accel_y;
    frame.imu.accel_z      = imu->accel_z;
    frame.imu.gyro_x       = imu->gyro_x;
    frame.imu.gyro_y       = imu->gyro_y;
    frame.imu.gyro_z       = imu->gyro_z;
    frame.imu.timestamp_us = imu->timestamp_us;

    // Fill wheel sub-message
    frame.has_wheel = true;
    frame.wheel.speed_fl     = wheel->speed_fl;
    frame.wheel.speed_fr     = wheel->speed_fr;
    frame.wheel.speed_rl     = wheel->speed_rl;
    frame.wheel.speed_rr     = wheel->speed_rr;
    frame.wheel.timestamp_us = wheel->timestamp_us;

    // Encode — writes TLV bytes into buffer
    bool ok = pb_encode(&stream, SensorFrame_fields, &frame);
    if (!ok) {
        printk("nanopb encode error: %s\n", PB_GET_ERROR(&stream));
        return 0;
    }

    return (uint16_t)stream.bytes_written;
}
```

---

## Decoding (Jetson / Python side)

```python
# Generate Python bindings from same .proto file
# protoc --python_out=. sensor_frame.proto
# → sensor_frame_pb2.py

from sensor_frame_pb2 import SensorFrame

def decode_frame(raw_bytes: bytes) -> SensorFrame | None:
    frame = SensorFrame()
    try:
        frame.ParseFromString(raw_bytes)
    except Exception as e:
        print(f"Decode error: {e}")
        return None
    return frame

# Usage
frame = decode_frame(payload)
if frame and frame.HasField('imu'):
    print(f"accel_x = {frame.imu.accel_x:.3f}")
    print(f"timestamp = {frame.imu.timestamp_us} us")

# Missing fields return defaults (0 for numbers) — never crashes
print(frame.wheel.speed_fl)  # 0.0 if wheel not in frame — safe
```

---

## Versioning — How It Actually Works

```
Firmware v1 sends:   SensorFrame { seq, imu }
Firmware v2 sends:   SensorFrame { seq, imu, wheel, temperature }

Jetson software v1 receives v2 frame:
  - Decodes seq ✓
  - Decodes imu ✓
  - Sees wheel field → unknown → silently skips ✓
  - Sees temperature field → unknown → silently skips ✓
  - No crash, no error

Firmware v2 receives from Jetson v1 config command (missing new fields):
  - Missing fields decoded as zero/default ✓
```

**Rules:**
1. Never remove a field from .proto — just stop using it in code
2. Never reuse a field number
3. Never change a field's type
4. Always add new fields with new numbers
5. Test with old receiver + new sender AND new receiver + old sender

---

## Enabling nanopb in Zephyr (CMakeLists.txt + Kconfig)

```kconfig
# prj.conf
CONFIG_NANOPB=y
```

```cmake
# CMakeLists.txt
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(my_app)

# Generate nanopb C files from .proto
nanopb_generate_cpp(PROTO_SRCS PROTO_HDRS proto/sensor_frame.proto)

target_sources(app PRIVATE
    src/main.c
    src/packer.c
    ${PROTO_SRCS}
)

target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}  # where generated .pb.h goes
)
```

---

## Sizing — How Big Will My Packet Be?

Quick estimate:
```
Each float field:    5 bytes (1 tag + 4 value)
Each uint64 varint:  2-10 bytes (depends on value size)
Each sub-message:    2 bytes overhead (tag + length) + contents
Sequence uint32:     1-5 bytes

SensorFrame example:
  seq (uint32=5):      2 bytes
  imu (sub-msg):       2 + (6 floats × 5) + timestamp ~= 34 bytes
  wheel (sub-msg):     2 + (4 floats × 5) + timestamp ~= 26 bytes
  Total:              ~64 bytes
```

For your output buffer size — always allocate `1.5× the estimated size` plus `pb_get_encoded_size()` for safety.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot `_init_zero` | Random garbage in unset fields | Always use `SensorFrame frame = SensorFrame_init_zero` |
| Buffer too small | `pb_encode` returns false | Increase buffer or check `pb_get_encoded_size` first |
| Wrong .pb.h version | Decoder crashes with bad access | Regenerate from same .proto on both sides |
| Reused field number | Old data decoded as new field | Never reuse — use reserved keyword in proto |
| Pointer in message | Compile error or corrupt data | nanopb does not support raw pointers; use `bytes` with callback |
