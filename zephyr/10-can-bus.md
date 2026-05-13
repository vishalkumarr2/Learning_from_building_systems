# CAN Bus — Controller Area Network

## What Is CAN?

CAN is a **differential, multi-master, broadcast serial protocol** invented by Bosch in 1983 for automotive use.

"Controller Area Network" — originally designed so ECUs (engine control units) in cars could talk to each other without a central computer.

Used for: wheel speed sensors, motor controllers (BLDC, servo), steering modules, brake controllers, battery management systems, industrial sensors — anything that needs **high reliability** and **long cables** in noisy electrical environments.

In our robot: CAN is how the STM32 reads **wheel encoders** and **motor controller feedback** at 100Hz.

---

## Physical Layer — Differential Signaling

```
STM32 CAN             Wheel Encoder           Motor Controller
────────────────────────────────────────────────────────────────
CAN_H ──────────────────────────────────────────────────────► 120Ω ◄─
                                                              terminator
CAN_L ──────────────────────────────────────────────────────► 120Ω ◄─
                                                              terminator
```

**CAN_H** and **CAN_L** are always used as a pair. The signal is the **voltage difference** between them:

```
Recessive (logical 1):  CAN_H = 2.5V,  CAN_L = 2.5V   → difference = 0V
Dominant  (logical 0):  CAN_H = 3.5V,  CAN_L = 1.5V   → difference = 2V
```

Why differential? If a motor injects noise onto the cable, **both wires are affected equally**. The receiver sees the difference, so noise cancels out. A single-ended wire (like UART) would be corrupted.

**Termination resistors** (120Ω) at each end of the bus prevent signal reflections at high speeds.

---

## CAN Frame Format

Every message on CAN has this format:

```
┌─────┬───────────────┬─────┬──────────────────────┬─────┬──────┐
│ SOF │  ID (11 bits) │ RTR │  DLC (4 bits)         │DATA │ CRC  │
│  1  │  (or 29 bits) │  1  │  0-8 = bytes of data  │0-64B│ 15+1 │
└─────┴───────────────┴─────┴──────────────────────┴─────┴──────┘
 SOF: Start of Frame
 ID:  Arbitration ID (who is this message from / what type)
 RTR: Remote Transmission Request  (0=data, 1=request)
 DLC: Data Length Code
 CRC: Cyclic Redundancy Check (error detection)
```

**Standard CAN** (CAN 2.0A): 11-bit ID → 2048 unique message IDs  
**Extended CAN** (CAN 2.0B): 29-bit ID → 536 million unique message IDs  
**CAN FD**: Up to 64 bytes data (vs 8 bytes in classic), up to 8 Mbps

In classic CAN, **max 8 bytes per frame**. For more data, you send multiple frames (CANopen, J1939 handle this).

---

## Bus Arbitration — How Multiple Masters Coexist

CAN has no master. Any node can transmit whenever the bus is idle. If two nodes start at the same time:

```
Node A transmits ID: 0b00001000010 = 0x042
Node B transmits ID: 0b00000100001 = 0x021

Bit 10: A=0, B=0  same, both continue
Bit  9: A=0, B=0  same, both continue
...
Bit  6: A=1, B=0  ← A transmits recessive (1), B transmits dominant (0)
```

**Dominant (0) wins**. Node A sees the bus doesn't match what it sent → **A stops and waits**.  
Node B wins and continues transmission. **Node with lower ID (higher priority) always wins**.

Zero collisions. Zero data corruption. Node A automatically retransmits when bus is free.

**Why this matters**: in a robot, you can assign IDs by priority:
- ID 0x001: Emergency stop → always gets through first
- ID 0x010: Wheel speed → high priority  
- ID 0x100: Status heartbeat → low priority, can be delayed

---

## Speeds

| Speed | Max Cable Length | Use |
|---|---|---|
| 1 Mbps | ~25m | Industrial automation, robotics |
| 500 kbps | ~100m | Automotive (most common) |
| 250 kbps | ~250m | Agriculture, slow processes |
| 125 kbps | ~500m | Building automation |

At 1 Mbps, **one 8-byte frame takes ~120µs**. Sending wheel speed at 100Hz = 1.2% bus utilization.

---

## CAN in Zephyr

### Devicetree

```dts
/* app.overlay */
&can1 {
    status = "okay";
    pinctrl-0 = <&can1_default>;
    pinctrl-names = "default";
    bus-speed = <1000000>;  /* 1 Mbps */
    sjw = <1>;
    sample-point = <875>;  /* 87.5% sample point */
};
```

### Sending a CAN frame

```c
#include <zephyr/drivers/can.h>

static const struct device *can_dev = DEVICE_DT_GET(DT_NODELABEL(can1));

void can_send_wheelspeed(float left_mps, float right_mps)
{
    /* Pack 2x float into 8 bytes — or use fixed-point for precision */
    int16_t left_fixed  = (int16_t)(left_mps  * 100);   /* 0.01 m/s resolution */
    int16_t right_fixed = (int16_t)(right_mps * 100);

    struct can_frame frame = {
        .flags = 0,  /* standard ID, data frame */
        .id  = 0x010,
        .dlc = 4,
        .data = {
            (left_fixed >> 8) & 0xFF,
            left_fixed & 0xFF,
            (right_fixed >> 8) & 0xFF,
            right_fixed & 0xFF,
        },
    };

    int rc = can_send(can_dev, &frame, K_MSEC(10), NULL, NULL);
    if (rc != 0) {
        LOG_ERR("CAN send failed: %d", rc);
    }
}
```

### Receiving CAN frames with a filter

```c
static struct can_filter wheel_filter = {
    .flags = 0,
    .id    = 0x010,       /* accept only ID 0x010 */
    .mask  = CAN_STD_ID_MASK  /* all 11 bits must match */
};

static void wheel_rx_callback(const struct device *dev,
                               struct can_frame *frame,
                               void *user_data)
{
    /* Called from CAN interrupt/thread — keep short */
    int16_t left_fixed  = (frame->data[0] << 8) | frame->data[1];
    int16_t right_fixed = (frame->data[2] << 8) | frame->data[3];

    struct wheel_msg msg = {
        .left_mps  = left_fixed  / 100.0f,
        .right_mps = right_fixed / 100.0f,
        .ts = k_uptime_get_32(),
    };

    zbus_chan_pub(&wheel_chan, &msg, K_NO_WAIT);
}

void can_init(void)
{
    /* Start CAN controller */
    can_start(can_dev);

    /* Register filter — only frames matching filter call our callback */
    int filter_id = can_add_rx_filter(can_dev, wheel_rx_callback,
                                      NULL, &wheel_filter);
    if (filter_id < 0) {
        LOG_ERR("Failed to add CAN filter: %d", filter_id);
    }
}
```

### Reading wheel encoder — real robot example

```c
/* Motor controller broadcasts ID=0x201 containing:
 *   bytes 0-1: actual position (int16, encoder ticks)
 *   bytes 2-3: actual velocity (int16, RPM)
 *   bytes 4-5: actual current  (int16, mA)
 *   bytes 6-7: temperature     (int16, 0.1°C)
 *
 * This is inspired by Roboteq MDC2450 protocol
 */

struct can_filter motor_filter = {
    .id   = 0x201,
    .mask = CAN_STD_ID_MASK,
};

static void motor_feedback_cb(const struct device *dev,
                               struct can_frame *frame,
                               void *user_data)
{
    if (frame->dlc != 8) return;

    int16_t ticks = (int16_t)sys_get_be16(&frame->data[0]);
    int16_t rpm   = (int16_t)sys_get_be16(&frame->data[2]);
    int16_t ma    = (int16_t)sys_get_be16(&frame->data[4]);
    int16_t temp  = (int16_t)sys_get_be16(&frame->data[6]);

    /* Convert */
    float velocity_mps = rpm / 60.0f * WHEEL_CIRCUMFERENCE_M;
    float current_a    = ma / 1000.0f;
    float temp_c       = temp / 10.0f;

    /* Overheat check */
    if (temp_c > 80.0f) {
        LOG_WRN("Motor hot: %.1f°C", (double)temp_c);
    }

    /* Publish to ZBus */
    struct motor_state msg = { .vel_mps = velocity_mps, .current_a = current_a };
    zbus_chan_pub(&motor_chan, &msg, K_NO_WAIT);
}
```

---

## CAN Error Handling

CAN has built-in **Error Confinement** — nodes that keep causing errors automatically shut down:

```
Error Active   → Error Passive   → Bus Off
(normal)           (TEC > 127)    (TEC > 255, node goes silent)
                 (sends passive   (must be manually restarted)
                  error flags)
```

TEC = Transmit Error Counter. Every consecutive TX error increments it.

In Zephyr, monitor state:

```c
static void can_state_change_cb(const struct device *dev,
                                 enum can_state state,
                                 struct can_bus_err_cnt err,
                                 void *user_data)
{
    LOG_INF("CAN state: %d TEC=%d REC=%d",
            state, err.tx_err_cnt, err.rx_err_cnt);

    if (state == CAN_STATE_BUS_OFF) {
        LOG_ERR("CAN bus off! Recovering...");
        can_recover(dev, K_MSEC(100));
    }
}

can_set_state_change_callback(can_dev, can_state_change_cb, NULL);
```

---

## CAN vs I2C vs SPI

| | CAN | I2C | SPI |
|---|---|---|---|
| Wires | 2 (differential) | 2 (single-ended) | 4+ |
| Speed | Up to 1 Mbps (8 Mbps FD) | Up to 1 MHz | Up to 50+ MHz |
| Distance | ~25m at 1 Mbps | <1m | <1m |
| Error detection | CRC + ACK + bit stuffing | ACK only | None |
| Multi-master | Yes (arbitration) | Yes (but rare) | No (one master) |
| Noise immunity | Excellent (differential) | Poor | Poor |
| Use in robot | Wheel encoders, motor control | Onboard sensors | Fast IMU, camera |

**Rule of thumb**: Use CAN when cables are long, environment is noisy, or nodes need to be hot-pluggable.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Missing termination resistors | Bus works at 1 node, fails with >2 | Add 120Ω at each end |
| CAN_H/CAN_L swapped | Nothing works | Check wiring polarity |
| Only one terminator | Works initially, fails at higher speeds | Both ends must have 120Ω |
| Not starting CAN controller | `-ENETDOWN` on send | Call `can_start()` before use |
| Blocking in rx callback | Other frames dropped | Publish to ZBus and return immediately |
| Not filtering by ID | CPU overwhelmed by all frames | Always install an rx filter |
