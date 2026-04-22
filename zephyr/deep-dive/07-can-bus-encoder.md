# 07 — CAN Bus: Receive Encoder Frames
### STM32H743ZI2 · Zephyr CAN API · SN65HVD230 transceiver · USB-CAN host verification

**Status:** 🟡 HARDWARE-GATED  
**Prerequisite:** Session `06` milestone complete  
**Hardware required:** 2× SN65HVD230 CAN transceiver breakouts · 4× 120Ω resistors · USB-CAN adapter (CANable or PCAN-USB)  
**Unlocks:** `08-uart-gps-nmea.md`  
**Time budget:** ~8 hours  
**Mastery plan:** Project 5

---

## Goal of This Session

Receive CAN frames from a second CAN node (simulated with `cansend` on the host via USB-CAN adapter), parse wheel velocity, and log the value at 100Hz. Verify with `candump` that frames are on the bus.

**Milestone**: `candump can0` on the host shows frames; STM32 logs `wheel_vel=1.23` updating live. Change the velocity with `cansend` → STM32 value changes within one CAN frame.

---

## Hardware Safety: CAN Transceiver Selection

**Use SN65HVD230, NOT TJA1050.**

The TJA1050 requires 5V VCC and its logic input threshold requires >3.5V. A 3.3V STM32 GPIO output is borderline — it may work at room temperature and fail in summer. The SN65HVD230 is designed for 3.3V systems. This is not a preference — it is a hardware compatibility requirement.

```
STM32H743           SN65HVD230 (Node A)     Bus          SN65HVD230 (Node B)    USB-CAN
─────────────       ─────────────────────   ─────────    ──────────────────     ───────
PD0 (CAN1_RX) ─── RX  │                   CANH ───────── CANH                   CANH
PD1 (CAN1_TX) ─── TX  │                   CANL ───────── CANL                   CANL
3.3V ──────────── VCC │                   120Ω ─ (term)── 120Ω ─ (term)
GND ───────────── GND │
                      └── RS pin → GND (enables normal mode; HIGH = slope control)
```

**Termination:** Place 120Ω resistors at each physical end of the bus. With 2 nodes + USB-CAN, the "ends" are: one resistor at the STM32 transceiver, one at the USB-CAN adapter. Measure with multimeter across CANH/CANL with power off: should read ~60Ω (two 120Ω in parallel).

---

## Theory: CAN Frame Format

```
SOF  ID[10:0]  RTR  IDE  r0  DLC[3:0]  DATA[0..7]  CRC[14:0]  CRC_DEL  ACK  ACK_DEL  EOF
 1     11       1    1    1     4        0–64 bits     15          1       1     1       7 bits
```

CAN 2.0A uses 11-bit standard IDs (0x000–0x7FF). CAN 2.0B uses 29-bit extended IDs.

**ID filtering**: the CAN controller filters frames by ID before the CPU sees them. Only frames matching your filter arrive in software. Set a filter that accepts only the encoder broadcast ID (e.g., 0x100).

**Bitrate**: both nodes must agree. Use **500kbps** for initial bringup — widely supported, easy to verify. Requires: prescaler + timing segments configured correctly. Zephyr's CAN devicetree handles this; you specify `bitrate = <500000>` in the overlay.

---

## Wiring

```
STM32 Nucleo-H743ZI2          SN65HVD230 Breakout
────────────────────          ──────────────────────
PD0 (CN9 pin 25) ──────────── RX
PD1 (CN9 pin 27) ──────────── TX
3.3V (CN6 pin 4) ──────────── VCC
GND  (CN6 pin 6) ──────────── GND
                               RS → GND (low slope / normal mode)
                               CANH → bus CANH
                               CANL → bus CANL

120Ω across CANH/CANL at STM32 end of bus
120Ω across CANH/CANL at USB-CAN adapter end of bus

Saleae Logic 8:
  Ch7 → CANH (differential pair — you won't decode CAN level directly,
               but you'll see dominant/recessive transitions)
  Or: use the CAN Analyzer in Logic 2 (differential, CANH+/CANL-).
```

---

## Pre-Code Verification: `candump` Before Zephyr

Before writing any Zephyr code, verify the USB-CAN adapter and bus wiring:

```bash
# On host Linux
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
candump can0 &

# Send a test frame from host → should see it in candump
cansend can0 100#AABBCCDD01020304
# candump output: can0  100   [8]  AA BB CC DD 01 02 03 04

# Stop
sudo ip link set can0 down
```

If `candump` shows nothing after `cansend`, check:
1. Termination resistors present and correct value
2. CANH and CANL not swapped
3. Both nodes have same bitrate
4. 60Ω across CANH/CANL measured with multimeter (power off)

**Only proceed to Zephyr code once `candump` confirms the bus is working.**

---

## Zephyr CAN Slave Application

### Devicetree Overlay

```dts
/* boards/nucleo_h743zi2.overlay */
&can1 {
    status = "okay";
    pinctrl-0 = <&can1_rx_pd0 &can1_tx_pd1>;
    pinctrl-names = "default";
    bitrate = <500000>;
    bus-speed-data = <500000>;  /* only needed for CANFD */
};
```

### prj.conf

```kconfig
CONFIG_CAN=y
CONFIG_CAN_INIT_PRIORITY=80
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_LOG=y
CONFIG_SHELL=y
```

### Main Application

```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/can.h>
#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(can_rx, LOG_LEVEL_DBG);

#define CAN_NODE     DT_NODELABEL(can1)
#define ENCODER_ID   0x100   /* matches cansend command above */
#define RX_QUEUE_LEN 8

/* Queue: CAN ISR → processing thread */
CAN_MSGQ_DEFINE(can_msgq, RX_QUEUE_LEN);

/* Frame format sent by encoder node:
   Bytes 0–1: uint16_t left_velocity_mms  (mm/s × 10, signed)
   Bytes 2–3: uint16_t right_velocity_mms
   Bytes 4–7: uint32_t timestamp_ms
*/
typedef struct {
    int16_t  left_vel;     /* mm/s × 10 */
    int16_t  right_vel;
    uint32_t timestamp_ms;
} encoder_frame_t;

static float last_wheel_vel = 0.0f;

void main(void) {
    const struct device *can_dev = DEVICE_DT_GET(CAN_NODE);

    if (!device_is_ready(can_dev)) {
        LOG_ERR("CAN device not ready");
        return;
    }

    /* Start CAN controller */
    int ret = can_start(can_dev);
    if (ret < 0 && ret != -EALREADY) {
        LOG_ERR("can_start failed: %d", ret);
        return;
    }

    /* Add RX filter: accept only ID 0x100, exact match (mask=0x7FF) */
    struct can_filter filter = {
        .flags   = 0,                 /* standard 11-bit ID */
        .id      = ENCODER_ID,
        .mask    = CAN_STD_ID_MASK,   /* 0x7FF — exact match */
    };

    int filter_id = can_add_rx_filter_msgq(can_dev, &can_msgq, &filter);
    if (filter_id < 0) {
        LOG_ERR("can_add_rx_filter failed: %d", filter_id);
        return;
    }

    LOG_INF("CAN RX started, filter id=%d, waiting for 0x%03X", filter_id, ENCODER_ID);

    struct can_frame frame;
    while (1) {
        /* Block until frame arrives */
        ret = k_msgq_get(&can_msgq, &frame, K_FOREVER);
        if (ret < 0) {
            LOG_ERR("k_msgq_get error: %d", ret);
            continue;
        }

        if (frame.dlc < 8) {
            LOG_WRN("Short frame: DLC=%d", frame.dlc);
            continue;
        }

        encoder_frame_t enc;
        memcpy(&enc, frame.data, sizeof(enc));

        float left_mms  = enc.left_vel  * 0.1f;
        float right_mms = enc.right_vel * 0.1f;
        float wheel_vel = (left_mms + right_mms) / 2.0f;  /* average */
        last_wheel_vel  = wheel_vel;

        LOG_INF("CAN 0x%03X: left=%.1f right=%.1f avg=%.1f mm/s ts=%u",
                frame.id, left_mms, right_mms, wheel_vel, enc.timestamp_ms);
    }
}
```

---

## Sending Test Frames from the Host

```bash
# Encode: left=1230 mm/s (12.3 m/s), right=1230 mm/s, ts=5000ms
# int16 1230 little-endian = CE 04; uint32 5000 little-endian = 88 13 00 00
cansend can0 100#CE04CE04881300 00

# Try different velocities
cansend can0 100#0000000000000000   # both stopped
cansend can0 100#FFFF000000000000   # left=-1 mm/s (braking)
```

**Watch STM32 shell:** `wheel_vel` should change within 1 frame of each `cansend`.

---

## Error Detection: CAN Bus-Off

If there's a wiring problem, the CAN controller may enter **bus-off** state:

```c
/* Add error callback to detect bus-off */
void can_error_cb(const struct device *dev, enum can_state state,
                  struct can_bus_err_cnt err_cnt, void *user_data) {
    LOG_ERR("CAN state: %d, TX err=%d, RX err=%d",
            state, err_cnt.tx_err_cnt, err_cnt.rx_err_cnt);
    if (state == CAN_STATE_BUS_OFF) {
        LOG_ERR("BUS-OFF — likely: bitrate mismatch, missing termination, or wiring fault");
        /* Recovery: */
        can_recover(dev, K_SECONDS(1));
    }
}

/* Register before can_start: */
can_set_state_change_callback(can_dev, can_error_cb, NULL);
```

**Bus-off diagnosis checklist:**
1. Measured 60Ω across CANH/CANL with multimeter? (power off)
2. `candump can0` shows anything on the host? If not → transceiver or wiring fault
3. Bitrate matches? STM32 `bitrate = <500000>` and host `ip link set can0 type can bitrate 500000`
4. CANH and CANL not swapped? (CANH is typically the higher-voltage wire)

---

## Milestone Checklist

- [ ] `candump can0` on host shows loopback: `cansend` frames appear
- [ ] Multimeter reads ~60Ω across CANH/CANL (power off)
- [ ] STM32 logs `wheel_vel=X.X mm/s` for each `cansend` frame
- [ ] Bus-off error callback never fires during 5-minute run
- [ ] STM32 shell `can1 stats` shows zero error frames
- [ ] Velocity changes live: `cansend can0 100#0000...` → STM32 logs `0.0 mm/s`

---

## Pre-Read for Session 8

Before `08-uart-gps-nmea.md`:
1. Read `zephyr/11_uart.md` in full — UART IRQ callback and ring buffer pattern
2. NMEA 0183 sentence format: `$GNGGA,hhmmss.ss,lat,N,lon,E,fix,sats,hdop,alt,M,...*checksum`
3. `00-mastery-plan.md` "NMEA sentences arrive split across two UART interrupts"

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Bus Verification (pre-code)
- Multimeter reading (power off): ___Ω (target: ~60Ω)
- `candump` loopback worked: yes/no
- Time to first valid loopback: ___min

### Zephyr CAN
- Filter ID used: 0x___
- First received frame: ...
- Bus-off triggered: yes/no (cause if yes: ...)

### Issues
- ...
```
