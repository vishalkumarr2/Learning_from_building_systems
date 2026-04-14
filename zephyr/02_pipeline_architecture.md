# STM32 → Jetson Orin — Full Pipeline Architecture

## Overview

A sensor aggregation system running at **100Hz** (one complete data frame every 10ms):

- **STM32** (Zephyr RTOS) — real-time sensor hub
  - Reads IMU over I2C at 100Hz
  - Receives wheel speed over CAN (event-driven)
  - Aggregates via ZBus, serializes with nanopb
  - Sends over SPI as slave

- **Jetson Orin** (Linux + ROS2) — high-level compute node
  - SPI master — drives 100Hz clock
  - Deserializes nanopb frames
  - Publishes to ROS2 topics

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STM32 (Zephyr RTOS)                          │
│                                                                     │
│  ┌──────────┐  I2C    ┌─────────────┐                              │
│  │ IMU      │────────►│ imu_thread  │──► zbus_chan_pub(imu_chan)    │
│  │ ICM42688 │         │  (100Hz)    │                    │          │
│  └──────────┘         └─────────────┘                    │          │
│                                                           ▼          │
│  ┌──────────┐  CAN    ┌─────────────┐          ┌──────────────────┐│
│  │ Wheel    │────────►│ can_thread  │──────────►│  packer_thread   ││
│  │ encoders │         │  (event)    │           │  (100Hz, 10ms)   ││
│  └──────────┘         └─────────────┘           │  nanopb encode   ││
│                                                  │  buffer swap     ││
│  ┌──────────┐  UART   ┌─────────────┐           └────────┬─────────┘│
│  │ GPS/etc  │────────►│ uart_thread │──────────►         │          │
│  └──────────┘         └─────────────┘                    │          │
│                                                           ▼          │
│                                                  ┌──────────────────┐│
│                                                  │  SPI slave + DMA ││
│                                                  │  double buffer   ││
│                                                  └────────┬─────────┘│
└───────────────────────────────────────────────────────────┼──────────┘
                                                            │ SPI
                                                            │ 10MHz
                                                            │ 10ms frames
┌───────────────────────────────────────────────────────────┼──────────┐
│                      Jetson Orin (Linux + ROS2)            │          │
│                                                            ▼          │
│                                                  ┌──────────────────┐│
│                                                  │  spidev master   ││
│                                                  │  100Hz timer     ││
│                                                  └────────┬─────────┘│
│                                                           │           │
│                                                           ▼           │
│                                                  ┌──────────────────┐│
│                                                  │  nanopb decode   ││
│                                                  └────────┬─────────┘│
│                                                           │           │
│                              ┌────────────────────────────┤           │
│                              ▼                            ▼           │
│                    /imu/raw (100Hz)          /wheel_speed (100Hz)     │
│                    sensor_msgs/Imu           geometry_msgs/Twist      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Per 10ms Frame

```
t = 0ms   : k_msleep(10) expires — IMU thread wakes
t = 0.5ms : I2C read completes → zbus_chan_pub(&imu_chan, &msg)
t = 0.5ms : CAN event may arrive any time → zbus_chan_pub(&wheel_chan, &msg)
t = 1.0ms : packer_thread wakes (periodic 10ms)
t = 1.0ms : zbus_chan_read(&imu_chan)  — grab latest IMU
t = 1.0ms : zbus_chan_read(&wheel_chan) — grab latest wheel data
t = 1.2ms : nanopb pb_encode() → ~150 bytes
t = 1.5ms : buffer swap — idle buffer becomes DMA active buffer
            (3.5ms safety margin before Jetson reads)
t = 5.0ms : Jetson 100Hz ROS2 timer fires
t = 5.0ms : Jetson asserts CS low
t = 5.0ms : CS ISR on STM32 — DMA already loaded, starts shifting
t = 5.05ms: SPI transfer complete (~50µs for 150 bytes @ 10MHz)
t = 5.1ms : Jetson nanopb decode — ~0.1ms (Python)
t = 5.2ms : rclpy publisher.publish() — ROS2 topic updated
t = 10ms  : cycle repeats
```

**3.5ms safety margin** between STM32 finishing the frame and Jetson reading it.

---

## Thread Architecture on STM32

| Thread | Priority | Stack | Wakeup | Job |
|---|---|---|---|---|
| SPI CS ISR | 2 (highest) | ISR stack | CS GPIO falling edge | Reload DMA pointer |
| CAN RX | 3 | 512B | CAN frame interrupt | Decode → zbus pub |
| IMU (I2C) | 5 | 1024B | k_msleep(10) | Read sensor → zbus pub |
| Other sensors | 6 | 1024B | k_msleep(10) | Read → zbus pub |
| Packer | 8 | 2048B | k_msleep(10) | Read zbus → nanopb encode → buffer swap |
| Logger | 10 (lowest) | 1024B | zbus subscriber | Log to flash/UART |

> Lower priority number = higher priority in Zephyr.
> Packer is lower priority than sensors so sensors always finish writing before packer reads.

---

## Why These Design Choices?

### Why SPI master on Jetson (not STM32)?

- Jetson controls the clock → deterministic 100Hz on the ROS2 side
- STM32 just needs data ready before CS goes low — easier to guarantee with double buffering
- If STM32 were master: needs a "ready" handshake GPIO, adds 2-way complexity

### Why ZBus instead of shared globals + mutex?

- Thread-safe by design — no manual locking in application code
- Multiple threads can subscribe to same channel (packer + logger both read IMU)
- Decouples sensor drivers from packer completely — swap out IMU driver without touching packer

### Why nanopb instead of raw structs?

- Endianness handled automatically
- Versioning: add a field → old receivers ignore it safely
- Self-framing: receiver doesn't need to know struct layout out-of-band
- Compact encoding: small numbers use fewer bytes (varint)

### Why double buffering for SPI?

- Jetson clocks data out immediately when CS goes low
- CPU and DMA cannot safely write to the same buffer at the same time
- Double buffer: DMA reads buffer A while CPU writes buffer B → zero race condition

---

## Protobuf Schema (the shared contract)

```protobuf
// sensor_frame.proto — shared between STM32 firmare and Jetson software
syntax = "proto3";

message ImuData {
    float accel_x       = 1;
    float accel_y       = 2;
    float accel_z       = 3;
    float gyro_x        = 4;
    float gyro_y        = 5;
    float gyro_z        = 6;
    uint64 timestamp_us = 7;
}

message WheelData {
    float speed_fl      = 1;
    float speed_fr      = 2;
    float speed_rl      = 3;
    float speed_rr      = 4;
    uint64 timestamp_us = 5;
}

message SensorFrame {
    uint32 seq    = 1;
    ImuData imu   = 2;
    WheelData wheel = 3;
}
```

Field numbers (1, 2, 3...) are permanent — never reuse a retired number.

---

## Wire Frame Format (SPI packet)

```
Byte 0:        0xAA          — sync byte (detect desync)
Byte 1-2:      length        — big-endian uint16, payload size
Byte 3..N+2:   nanopb payload — SensorFrame encoded bytes
Byte N+3..N+4: CRC16-CCITT   — integrity check

Total: 5 + payload bytes
Typical payload: ~120-200 bytes depending on how many fields present
```

The sync byte lets the Jetson detect if it missed a byte and needs to re-lock.
