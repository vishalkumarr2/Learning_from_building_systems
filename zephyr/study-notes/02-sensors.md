# Zephyr Sensors — Study Notes
### Projects 4–6: I2C IMU · CAN Encoders · UART GPS
**Hardware:** STM32 Nucleo-H743ZI2 · ICM-42688-P · SN65HVD230 CAN transceiver · u-blox M8N GPS

> **Electronics prerequisites:** If open-drain, pull-up calculations, or differential signaling feel unfamiliar, read these first:
> - I2C physics → `electronics/06-i2c-deep-dive.md` (open-drain, pull-up calculation, wired-AND)
> - CAN physics → `electronics/07-can-deep-dive.md` (differential signaling, arbitration, termination)
> - UART protocol → `electronics/04-uart-serial-deep-dive.md` (frame format, baud rate, RS-232 vs TTL)

---

## PART 1 — ELI15 Concept Explanations

---

### 1.1 Why I2C Uses Only 2 Wires — Open-Drain and the Wired-AND Trick

SPI needs four wires: clock, data-out, data-in, and a chip-select per slave. Add three sensors and you have ten wires. I2C does the same job with two wires regardless of how many sensors you add. The reason comes down to how the electrical lines work.

Each I2C wire (SDA and SCL) is an **open-drain** line. "Open-drain" means every device on the bus has a transistor that can only *pull the wire LOW* — down toward 0V. No device can actively push the wire HIGH. Instead, a pull-up resistor connected to VCC (typically 4.7 kΩ) constantly tries to pull both wires HIGH. The wire rests HIGH by default. When any device wants to signal a `0` bit, it fires its transistor and drags the wire LOW, overpowering the pull-up resistor.

The crucial property is: if *anyone* on the bus pulls LOW, the wire reads LOW, regardless of what everyone else is doing. This is the "wired-AND" trick — the wire only reads HIGH when **every** device is leaving it alone. This means multiple devices can share the same wire without needing separate data lines, because they communicate using addresses sent as bit patterns. The master sets the clock, calls a specific address, and only the slave with that matching address responds.

Compare this to SPI's **push-pull** outputs: an SPI device actively drives the wire HIGH by connecting it to VCC, or LOW by connecting it to GND. If two push-pull devices try to talk at once — one pushing HIGH and the other pushing LOW — you get a short circuit. That is why SPI devices must never share data lines without a multiplexer. I2C's open-drain design avoids this problem entirely; the worst that happens when two devices try to talk at once is that both see LOW and the one that wanted to send HIGH backs off.

---

### 1.2 The I2C Address Byte — 7 Bits, One R/W Bit, and the ACK

When the I2C master wants to talk to a specific slave, it sends one byte to start every transaction. This byte carries three pieces of information packed together, and understanding each one prevents hours of debugging.

**The 7-bit address** occupies the top seven bits (bits 7–1). The ICM-42688 IMU, for example, has two possible addresses: `0x68` (binary `1101000`) when its `AD0` pin is grounded, or `0x69` (binary `1101001`) when `AD0` is pulled high. The only difference between these two addresses is the last bit — which is exactly which pin you wire. This is how you put two identical IMUs on one bus: connect one `AD0` to GND, the other to VCC, and they become two different devices.

**The R/W bit** is bit 0 of the address byte. When it is `0`, the master is about to *write* data to the slave (for example, to configure a register). When it is `1`, the master wants to *read* data from the slave. The slave itself does not set this bit — the master controls it. This is why `i2c_write_read()` in Zephyr does two separate transactions internally: first a write (R/W=0) to send the register address, then a read (R/W=1) to receive the data.

**The ACK bit** is the ninth clock cycle after the 8-bit address+R/W byte. Here the slave takes control of SDA and pulls it LOW to say "yes, I'm here and ready." If nothing pulls it LOW (because no slave recognized the address, or the slave is broken, or the pull-up resistor is missing), SDA stays HIGH. This is a NACK — and Zephyr returns `-ENXIO` (no such device). Learning to see the ACK slot on a logic analyzer is the single fastest way to diagnose I2C problems.

---

### 1.3 Pull-Up Resistors — Why They're Required and Why 10 kΩ Fails at 400 kHz

Pull-up resistors on I2C are not optional. The lines are open-drain: they can only be driven LOW by devices. The pull-up is the *only* mechanism for driving them HIGH. Without pull-ups, the lines float at an indeterminate voltage when no device is pulling them down, and the bus simply does not work.

The value of the pull-up resistor matters more than most tutorials explain. The relationship is governed by the RC time constant of the line. Every I2C line has capacitance — the wires, PCB traces, and the input capacitance of each device all add up. A typical bus with three devices and 20 cm of PCB trace might have 50–100 pF of capacitance. When a device releases SDA (stops pulling LOW), the pull-up resistor and this capacitance form an RC circuit. The time it takes the voltage to rise from 0V to the HIGH threshold (around 0.7 × VCC = 2.3V on a 3.3V system) is approximately **2.2 × R × C**.

At 400 kHz (Fast mode), each clock half-period is 1.25 µs. The line must rise completely HIGH within that window. With C = 100 pF and R = 10 kΩ: rise time = 2.2 × 10,000 × 100×10⁻¹² = **2.2 µs**. That exceeds the 1.25 µs budget — the line never fully reaches HIGH before the next clock edge. The master misreads a `1` as a `0`. At low speeds (100 kHz, 5 µs half-period), the same 10 kΩ works fine, which is why your sensor might work at 100 kHz but fail silently at 400 kHz with no error codes — just corrupted data.

The fix is to lower the resistance. 4.7 kΩ at 100 pF gives a rise time of 1.03 µs — safely within budget. 2.2 kΩ gives 0.48 µs, which works at Fast-mode+ (1 MHz). The downside of lower resistance: higher current draw for the entire time the line is pulled LOW. At 4.7 kΩ and 3.3V, that is 0.7 mA per line per pull-LOW — usually acceptable. At 1 kΩ it becomes 3.3 mA per line, which begins to stress devices.

**Practical rule:** Use 4.7 kΩ for 100–400 kHz, 2.2 kΩ for 1 MHz. Check whether your breakout board already has pull-ups (many do). Parallel pull-ups reduce the effective resistance — two 4.7 kΩ in parallel = 2.35 kΩ, which will overdrive the bus.

---

### 1.4 CAN Bus Differential Signaling — Why It's Immune to Noise

Early automotive networks used a single wire for data. When a motor or ignition coil switched on nearby, it induced a voltage spike onto that wire. The receiver could not tell a legitimate `0` bit from a noise spike, so the data was corrupted. This was acceptable for simple systems but catastrophic when the data controlled brakes or engine injection.

CAN uses two wires (CAN_H and CAN_L) and the signal is the *difference* between them. In a recessive state (logical 1), both wires sit at 2.5V — the difference is 0V. In a dominant state (logical 0), CAN_H rises to 3.5V and CAN_L falls to 1.5V — the difference is 2.0V. The receiver measures only the difference, never the absolute voltage of either wire.

When a noise source — a motor, a welding spark, a power supply switching — induces a voltage onto the cable, it induces the *same* voltage onto both CAN_H and CAN_L simultaneously, because they are routed together as a twisted pair. If noise adds 0.5V to both wires: CAN_H goes from 3.5V to 4.0V, CAN_L goes from 1.5V to 2.0V. The difference remains 2.0V. The receiver sees: dominant state, logical `0`. The noise was completely cancelled out.

This is why CAN is used in cars, industrial robots, and warehouses — all environments with heavy motors, switching power supplies, and long cable runs. UART and I2C use single-ended signaling (voltage relative to GND) and are easily corrupted in these environments. CAN's differential nature makes it functionally immune to common-mode noise.

---

### 1.5 CAN Bus Arbitration — How Two Nodes Transmit Simultaneously Without Collision

In most protocols, if two devices try to talk at the same time, you get a collision and both messages are destroyed (Ethernet's CSMA/CD handles this with random backoff). CAN solves the collision problem in hardware, so lower-priority messages are silently pushed aside and the highest-priority message gets through intact on the first attempt.

CAN uses the same wired-AND physics as I2C. The bus line is pulled recessive (HIGH, logical `1`) by termination resistors. Any node can force it dominant (LOW, logical `0`) by driving the line. If two nodes write `1` (recessive) simultaneously, both see `1` on the bus — they agree. If one writes `0` (dominant) and the other writes `1` (recessive), the bus reads `0` — the dominant wins, and the node that sent `1` but sees `0` knows something overrode it.

Arbitration works by having both nodes transmit their message ID simultaneously, bit by bit, and continuously listen to what is actually on the bus. While a node's transmitted bit matches what it reads back, it continues. The instant a node transmits `1` but reads back `0`, it knows another node has a lower (higher priority) ID and is continuing. Node A immediately stops transmitting and schedules a retry. Node B never noticed the collision — it was winning every bit — and continues to completion. The winning message arrives on the bus intact. Node A retransmits as soon as the bus goes idle.

The implication for your robot: message IDs double as priorities. CAN ID `0x001` will always beat `0x100` in arbitration. Design your IDs deliberately: emergency stop at `0x001`, motor setpoints at `0x010`, sensor readings at `0x100`, status heartbeats at `0x200`. Under bus overload, the lower-priority status messages will be delayed, but safety-critical messages get through immediately.

---

### 1.6 CAN Termination Resistors — Why 120 Ω at Each End

A CAN bus is a transmission line. At high speeds (1 Mbps), the electrical signal travels along the cable as a wave. When the wave reaches the end of the cable, if there is nothing there to absorb it, it reflects back down the wire — the same way a wave in a bathtub bounces off the walls. The reflected wave interferes with new bits being sent on the wire, corrupting them.

The termination resistor at each end of the bus is sized to match the cable's **characteristic impedance**. A typical twisted-pair CAN cable has a characteristic impedance of approximately 120 Ω. When you place a 120 Ω resistor at each end, the arriving wave "sees" a resistor that looks exactly like more cable extending to infinity — it is fully absorbed, no reflection. The bus appears electrically as if it has no ends at all.

Without termination, signals reflect and bounce. At low speeds (125 kbps, 8 µs per bit), the reflections decay before the next bit arrives — the bus works fine. At 1 Mbps (1 µs per bit), reflections from the first bit are still bouncing when the third or fourth bit arrives. The receiver sees superimposed waveforms and cannot reliably determine the bit value. The error counters climb. Eventually the node enters Bus Off state and goes silent.

You can verify termination with a multimeter: power off the bus, then measure resistance between CAN_H and CAN_L. Two 120 Ω resistors in parallel measure **60 Ω**. If you measure 120 Ω, only one terminator is present. If you measure very high resistance or 40 Ω, you have one missing or an extra terminator somewhere. This two-second check has saved countless hours of debugging mysterious CAN failures.

---

### 1.7 UART — How It Works with No Clock Wire

Every serial protocol needs a way for the receiver to know when each bit begins and ends. I2C and SPI solve this with a dedicated clock wire: the clock master drives a square wave and the receiver samples data on each edge. UART solves it differently — both sides pre-agree on how fast bits will be transmitted, and each side uses its own crystal oscillator to count time.

Before any communication happens, the developer configures both devices to the same **baud rate** — bits per second. 115200 baud means each bit lasts exactly 1/115200 ≈ 8.68 µs. When the receiver sees the line go LOW (the start bit), it starts its internal timer. After 8.68 µs it samples bit 0. After another 8.68 µs it samples bit 1. It repeats for all 8 data bits, then checks for the stop bit (line returns HIGH). The entire byte is reconstructed from timing alone, with no clock wire.

The vulnerability is clock accuracy. If your microcontroller's oscillator runs at 0.5% faster than the theoretical frequency, by the eighth bit the receiver is sampling 4% (0.5% × 8) into the wrong time slot. At higher baud rates this matters more: at 9600 baud, there is 4.3× more time margin for drift than at 921600 baud. For embedded systems, a crystal oscillator (typically ±20 ppm, or 0.002%) is accurate enough for any standard baud rate. Internal RC oscillators (±1–3%) can work at 9600 baud but often fail at 115200 without calibration.

The start bit also serves as a **resynchronization** mechanism. Whenever the line goes idle (HIGH) then transitions LOW, the receiver resets its bit timer. This means even if the receiver drifts slightly between bytes, each new start bit corrects the drift. This is why UART is called "asynchronous" — the two sides never stay synchronized for more than one byte at a time, and they re-sync on every start bit.

---

### 1.8 NMEA Sentences — Format and Why They Arrive Split Across Interrupts

GPS modules output their data as plain ASCII text in a standardized format called NMEA 0183. Every sentence starts with `$`, ends with `\r\n`, contains comma-separated fields, and ends with `*HH` where HH is a two-digit hexadecimal checksum. A typical position sentence looks like:

```
$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\r\n
```

The fields are: time, status (A=valid, V=void), latitude, N/S, longitude, E/W, speed in knots, heading, date, variation, E/W, checksum. Each burst at 10 Hz is a group of several sentences back-to-back, amounting to perhaps 300–500 bytes per second.

The split-across-interrupts problem arises because UART receives one byte at a time, and your UART interrupt fires for every byte (or for every small burst if you have a hardware FIFO). A sentence of 80 bytes at 9600 baud takes 83 ms to arrive. Your interrupt fires dozens of times during that interval. In any given interrupt callback, you may receive the first 40 bytes of one sentence, the last 40 bytes, or even the middle of one sentence followed by the start of the next.

The correct mental model is: **your interrupt is a byte source, not a sentence source.** You must accumulate bytes into a buffer and only consider the buffer complete when you encounter `\n`. Never assume that one interrupt callback equals one complete sentence. Never call `sscanf` inside the interrupt — you call `sscanf` only after the full sentence delimiter has arrived. The ring buffer pattern exists precisely for this reason: each byte lands in the ring buffer immediately, and a separate reader thread polls the buffer for complete `\n`-terminated lines.

---

### 1.9 Ring Buffers — Why UART Uses Them

The UART receive interrupt fires many times per second, often while the CPU is in the middle of something else. Without a buffer, one of two bad things happens: either the interrupt waits for the CPU to finish (jitter, missed bytes), or the CPU has to poll the UART register fast enough to never miss a byte (burning 100% CPU).

A ring buffer is a fixed-size array with two pointers: a **write pointer** (where the next incoming byte gets placed) and a **read pointer** (where the next outgoing byte is taken from). The write pointer advances on each incoming byte (in the interrupt). The read pointer advances as your thread consumes bytes. The write pointer wraps around when it reaches the end — hence "ring." The buffer is empty when the pointers are equal, and full when the write pointer is one step behind the read pointer.

The key property: writing and reading are independent operations that never interfere with each other, as long as only one writer (the interrupt) and one reader (the thread) access the buffer. The interrupt does one thing: drop a byte at the write pointer and advance it. This takes a handful of nanoseconds. The interrupt returns immediately and never stalls waiting for the CPU to process data. Meanwhile, the reader thread wakes up periodically, drains bytes from the read pointer at its own pace, and assembles complete sentences.

Without a ring buffer at 115200 baud, you must read each byte within 87 µs or the next byte overwrites it in the UART's hardware FIFO (typically 4–16 bytes deep). With a 512-byte ring buffer, you have slack: even if your thread is preempted for 35 ms, the ring buffer holds the incoming bytes. You lose data only if the buffer fills completely — which is detectable: Zephyr's `ring_buf_put()` returns 0 when full, so you can log the overflow and know your buffer needs to be larger.

---

### 1.10 The IMU Data-Ready Interrupt — DRY Pin and ISR Handoff

The ICM-42688 IMU has a hardware output pin called INT1 (also called the DRDY pin — data ready). When a new accelerometer and gyroscope sample has been computed internally and written to the sensor's output registers, the sensor asserts this pin. Your microcontroller can monitor INT1 as a GPIO interrupt and read the sensor only when new data is actually available.

The alternative — polling the sensor every 10 ms with `k_msleep(10)` — has two problems. First, your 10 ms timer and the sensor's internal 10 ms ODR (output data rate) timer are not synchronized. Sometimes you read data that is 9.8 ms old, sometimes 0.2 ms old, creating up to 9.8 ms of timestamp jitter. Second, if the IMU is running at exactly 100 Hz but your thread wakes up 0.1 ms early, you read the previous sample. If it wakes up 0.1 ms late, there is a 10.2 ms gap and a 9.8 ms gap in the stream — visible jitter in a sensor fusion algorithm.

The DRDY interrupt eliminates both problems. The IMU raises INT1 exactly when new data is ready. Your ISR wakes immediately. The latency between data-ready and data-read is typically under 100 µs, dominated by I2C transfer time, not timer jitter.

The ISR itself should not do the I2C read — I2C transfers can take 300 µs and involve sleeping while waiting for GPIO edges. Sleeping in an ISR is forbidden in Zephyr (and most RTOSes). Instead, the ISR gives a semaphore: `k_sem_give(&imu_drdy_sem)`. A waiting thread (`k_sem_take(&imu_drdy_sem, K_FOREVER)`) wakes up in thread context where blocking I2C calls are allowed. This "ISR gives semaphore, thread does work" pattern is the standard handoff in all Zephyr sensor drivers.

---

## PART 2 — Annotated Code Reference

---

### 2.1 Devicetree I2C Node for ICM-42688

```dts
/* boards/nucleo_h743zi2.overlay */

/* &i2c1 refers to the I2C1 peripheral defined in the SoC DTS.
 * We are modifying it, not creating a new node. */
&i2c1 {
    status = "okay";                          /* enable this peripheral */
    pinctrl-0 = <&i2c1_scl_pb6 &i2c1_sda_pb7>;  /* alternate function mapping */
    pinctrl-names = "default";
    clock-frequency = <I2C_BITRATE_FAST>;    /* 400 kHz — use FAST, not STANDARD */
                                             /* STANDARD = 100 kHz — works, but slower */
                                             /* Do NOT write 400000 — use the macro */

    /* Child node: the ICM-42688 sits on this I2C bus.
     * Node name format: "compatible-name@address" — the @68 MUST match reg = <0x68> */
    imu: icm42688@68 {
        compatible = "invensense,icm42688p";
        reg = <0x68>;                         /* I2C address: 0x68 when AD0 pin is LOW */
                                              /* Change to <0x69> if AD0 is pulled HIGH */
        int-gpios = <&gpiob 3 GPIO_ACTIVE_HIGH>;  /* DRDY interrupt pin: PB3 */
                                              /* GPIO_ACTIVE_HIGH: INT1 goes HIGH on data-ready */
                                              /* Check ICM-42688 datasheet INT_CONFIG reg: */
                                              /* default is active-high pulse */
        label = "ICM42688";
    };
};
```

**Common mistakes:**
```dts
/* WRONG: integer frequency */
clock-frequency = <400000>;   /* syntactically valid but bypasses driver validation */

/* RIGHT: symbolic constant */
clock-frequency = <I2C_BITRATE_FAST>;

/* WRONG: address mismatch */
imu: icm42688@69 {
    reg = <0x68>;             /* Node address (after @) must match reg value */
}

/* RIGHT: they must match */
imu: icm42688@68 {
    reg = <0x68>;
}
```

---

### 2.2 WHO_AM_I Verification then Burst Read

```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(imu, LOG_LEVEL_DBG);

/* Get device from devicetree label — fails at compile time if label doesn't exist */
static const struct device *i2c_dev = DEVICE_DT_GET(DT_NODELABEL(i2c1));

#define IMU_ADDR         0x68   /* must match overlay reg = value */
#define ICM42688_WHO_AM_I_REG  0x75
#define ICM42688_WHO_AM_I_VAL  0x47   /* datasheet: WHO_AM_I returns 0x47 */
#define ICM42688_PWR_MGMT0     0x4E
#define ICM42688_ACCEL_XOUT_H  0x1F   /* first of 12 consecutive sensor bytes */

/* Scales depend on ODR/FSR register settings (not changed here) */
#define ACCEL_SCALE_G    (1.0f / 2048.0f)   /* ±16g range, 16-bit → g */
#define GYRO_SCALE_DPS   (1.0f / 16.4f)     /* ±2000 dps range → dps */

struct imu_sample {
    float accel_g[3];   /* x, y, z in g */
    float gyro_dps[3];  /* x, y, z in degrees/second */
    int64_t ts_us;      /* capture timestamp in microseconds */
};

/* Check WHO_AM_I — call this at init to verify the sensor is present and correct */
static int imu_verify_identity(void)
{
    uint8_t who_am_i = 0;
    uint8_t reg = ICM42688_WHO_AM_I_REG;

    /* i2c_write_read: performs a write (send register address)
     * followed by RESTART and a read (receive register value).
     * This is the standard I2C "register read" transaction.
     * Arguments: device, slave_addr, write_buf, write_len, read_buf, read_len */
    int rc = i2c_write_read(i2c_dev, IMU_ADDR,
                            &reg, 1,          /* write: 1-byte register address */
                            &who_am_i, 1);    /* read: 1-byte register value */

    if (rc != 0) {
        /* -ENXIO: slave did not ACK its address — wrong address or no pull-ups */
        /* -EIO:   slave ACK'd but SDA stuck — bus locked up, need recovery */
        LOG_ERR("WHO_AM_I read failed: %d", rc);
        return rc;
    }

    if (who_am_i != ICM42688_WHO_AM_I_VAL) {
        /* Got a response but wrong value — different sensor/revision */
        LOG_ERR("Unexpected WHO_AM_I: 0x%02x (expected 0x%02x)",
                who_am_i, ICM42688_WHO_AM_I_VAL);
        return -ENODEV;
    }

    LOG_INF("ICM-42688 identified. WHO_AM_I=0x%02x", who_am_i);
    return 0;
}

/* Burst read: reads all 12 sensor bytes in ONE I2C transaction.
 * The ICM-42688 auto-increments its internal register pointer on each byte read.
 * One transaction = 1 START + address + 12 data bytes + 1 STOP = ~300 µs at 400 kHz.
 * Compare to 12 separate reads: 12× the overhead, ~3.6 ms. Always burst-read. */
static int imu_read_sample(struct imu_sample *out)
{
    uint8_t raw[12];
    uint8_t start_reg = ICM42688_ACCEL_XOUT_H;

    int rc = i2c_write_read(i2c_dev, IMU_ADDR,
                            &start_reg, 1,   /* tell sensor: start at ACCEL_XOUT_H */
                            raw, 12);        /* read 12 bytes: 6 accel + 6 gyro */
    if (rc != 0) {
        return rc;
    }

    /* ICM-42688 outputs 16-bit big-endian signed integers.
     * raw[0] is HIGH byte of ACCEL_X, raw[1] is LOW byte.
     * (int16_t) cast is critical: without it, the shift/OR produces a uint16_t
     * and negative values (e.g. -9.8g downward) are interpreted as large positives. */
    out->accel_g[0] = (int16_t)((raw[0] << 8) | raw[1]) * ACCEL_SCALE_G;
    out->accel_g[1] = (int16_t)((raw[2] << 8) | raw[3]) * ACCEL_SCALE_G;
    out->accel_g[2] = (int16_t)((raw[4] << 8) | raw[5]) * ACCEL_SCALE_G;
    out->gyro_dps[0] = (int16_t)((raw[6]  << 8) | raw[7])  * GYRO_SCALE_DPS;
    out->gyro_dps[1] = (int16_t)((raw[8]  << 8) | raw[9])  * GYRO_SCALE_DPS;
    out->gyro_dps[2] = (int16_t)((raw[10] << 8) | raw[11]) * GYRO_SCALE_DPS;

    /* Capture timestamp as close to the read as possible.
     * k_ticks_to_us_near64 converts kernel ticks to microseconds.
     * This is more precise than k_uptime_get_32() which truncates to 32-bit ms. */
    out->ts_us = k_ticks_to_us_near64(k_uptime_ticks());

    return 0;
}

/* Full IMU initialization sequence */
static int imu_init(void)
{
    if (!device_is_ready(i2c_dev)) {
        LOG_ERR("I2C bus not ready");
        return -ENODEV;
    }

    /* Step 1: verify we're talking to the right chip */
    int rc = imu_verify_identity();
    if (rc != 0) return rc;

    /* Step 2: wake sensor from sleep.
     * PWR_MGMT0 value 0x0F: accel = low-noise mode, gyro = low-noise mode.
     * After reset, ICM-42688 is in sleep (0x00). YOU must write this register.
     * Without it, all sensor reads return 0 — looks like hardware failure. */
    uint8_t pwr_cmd[2] = { ICM42688_PWR_MGMT0, 0x0F };
    rc = i2c_write(i2c_dev, pwr_cmd, 2, IMU_ADDR);
    if (rc != 0) return rc;

    /* Step 3: wait for sensor to stabilize after mode change.
     * Datasheet specifies 200 µs minimum. Use 10 ms to be safe.
     * Skipping this sleep causes the first read to return zeros or garbage. */
    k_msleep(10);

    return 0;
}
```

---

### 2.3 I2C Bus Stuck Recovery — 9-Clock Bit-Bang Routine

```c
#include <zephyr/drivers/i2c.h>

/* When to use this:
 * - i2c_write_read returns -EIO on every call
 * - Logic analyzer shows SDA stuck LOW even when master is not transmitting
 * - Cause: power was lost while slave was in the middle of sending a byte —
 *          the slave is waiting for 1–8 more SCL pulses to complete its data byte
 *          before releasing SDA.
 *
 * Solution: manually clock SCL up to 9 times. After at most 9 clocks, any
 *           stuck slave will have finished its byte and released SDA.
 *           Then send a STOP condition to reset all state machines. */
static int imu_recover_bus(void)
{
    LOG_WRN("I2C bus stuck — attempting recovery");

    /* Zephyr provides i2c_recover_bus() which handles the bit-bang internally.
     * Call it first before attempting manual recovery. */
    int rc = i2c_recover_bus(i2c_dev);
    if (rc == 0) {
        LOG_INF("I2C bus recovered via i2c_recover_bus()");
        return 0;
    }

    /* If the automatic recovery failed (e.g. no bit-bang support on this driver),
     * fall back to a retry loop. On the second attempt, the slave may have had
     * time to time out internally and release SDA. */
    LOG_WRN("Automatic recovery failed (%d), retrying read...", rc);
    k_msleep(5);   /* give slave time to time out its stuck state */

    uint8_t who_am_i;
    uint8_t reg = ICM42688_WHO_AM_I_REG;
    rc = i2c_write_read(i2c_dev, IMU_ADDR, &reg, 1, &who_am_i, 1);
    if (rc != 0) {
        LOG_ERR("Recovery failed — bus still stuck: %d", rc);
        return rc;
    }

    LOG_INF("Bus recovered after retry. WHO_AM_I=0x%02x", who_am_i);
    return 0;
}

/* Wrapper that auto-recovers on -EIO */
static int imu_safe_read(struct imu_sample *out)
{
    int rc = imu_read_sample(out);
    if (rc == -EIO) {
        /* -EIO means bus is locked — slave holding SDA low */
        rc = imu_recover_bus();
        if (rc == 0) {
            /* Retry once after recovery */
            rc = imu_read_sample(out);
        }
    }
    return rc;
}
```

---

### 2.4 FDCAN Devicetree + `can_add_rx_filter()` + Frame Parsing

```dts
/* app.overlay — FDCAN (Flexible Data-rate CAN) on STM32H7 */

/* STM32H743 has FDCAN, not classic CAN. Zephyr supports both under the can API. */
&fdcan1 {
    status = "okay";
    pinctrl-0 = <&fdcan1_tx_pd1 &fdcan1_rx_pd0>;  /* check your board's pin mapping */
    pinctrl-names = "default";
    bus-speed = <1000000>;      /* 1 Mbps — must match ALL other nodes on the bus */
    sjw = <1>;                  /* synchronization jump width: 1 time quantum */
    sample-point = <875>;       /* 87.5% sample point — industry standard for 1 Mbps */
                                /* sample-point is in units of 0.1%, so 875 = 87.5% */
    /* For CAN FD data-phase speed (optional): */
    /* bus-speed-data = <4000000>; */
};
```

```c
#include <zephyr/drivers/can.h>
#include <zephyr/sys/byteorder.h>  /* for sys_get_be16() — safe endian conversion */

LOG_MODULE_REGISTER(can_wheel, LOG_LEVEL_DBG);

static const struct device *can_dev = DEVICE_DT_GET(DT_NODELABEL(fdcan1));

/* Motor controller CAN protocol (example, Roboteq-inspired):
 * ID 0x201: Motor feedback frame
 *   bytes 0-1: encoder position ticks (int16, big-endian)
 *   bytes 2-3: velocity RPM (int16, big-endian)
 *   bytes 4-5: current mA  (int16, big-endian)
 *   bytes 6-7: temperature (int16 × 10, big-endian, e.g. 235 = 23.5°C) */
#define MOTOR_FEEDBACK_ID  0x201
#define WHEEL_CIRCUMFERENCE_M  0.3142f   /* π × 0.1m diameter */
#define TICKS_PER_REV      4096          /* encoder resolution */

struct motor_state {
    float vel_mps;
    float current_a;
    float temp_c;
    int32_t ticks;
};

/* RX callback: called from CAN ISR or high-priority Zephyr CAN thread.
 * MUST be fast. The only work here: unpack integers, publish to ZBus, return.
 * Never call LOG from here (LOG is thread-context only in deferred mode). */
static void motor_feedback_cb(const struct device *dev,
                               struct can_frame *frame,
                               void *user_data)
{
    if (frame->dlc != 8) {
        /* Wrong payload size — protocol mismatch or wiring error */
        return;
    }

    /* sys_get_be16 reads 2 bytes as big-endian, returns uint16_t.
     * Cast to (int16_t) to restore sign. Without the cast, RPM=-500 
     * becomes 65036 and current math is silently wrong. */
    int16_t ticks = (int16_t)sys_get_be16(&frame->data[0]);
    int16_t rpm   = (int16_t)sys_get_be16(&frame->data[2]);
    int16_t ma    = (int16_t)sys_get_be16(&frame->data[4]);
    int16_t temp  = (int16_t)sys_get_be16(&frame->data[6]);

    struct motor_state msg = {
        .vel_mps   = (float)rpm / 60.0f * WHEEL_CIRCUMFERENCE_M,
        .current_a = (float)ma / 1000.0f,
        .temp_c    = (float)temp / 10.0f,
        .ticks     = ticks,
    };

    /* K_NO_WAIT: do not block in ISR/callback context.
     * If ZBus is full, we drop this sample — acceptable at 100Hz
     * because the next sample arrives in 10ms. Never K_FOREVER here. */
    zbus_chan_pub(&motor_chan, &msg, K_NO_WAIT);
}

void can_subsystem_init(void)
{
    if (!device_is_ready(can_dev)) {
        LOG_ERR("CAN device not ready");
        return;
    }

    /* MUST call can_start() before any send/receive operations.
     * Forgetting this is the most common cause of -ENETDOWN errors. */
    int rc = can_start(can_dev);
    if (rc != 0) {
        LOG_ERR("can_start failed: %d", rc);
        return;
    }

    /* Filter: only deliver frames with ID == 0x201 to our callback.
     * Without a filter (or with mask=0), EVERY frame on the bus wakes
     * our callback — including frames from unrelated nodes. At high bus
     * utilization this can saturate the CPU. */
    static struct can_filter motor_filter = {
        .flags = 0,                  /* standard 11-bit ID */
        .id    = MOTOR_FEEDBACK_ID,
        .mask  = CAN_STD_ID_MASK,    /* all 11 bits must match exactly */
    };

    int filter_id = can_add_rx_filter(can_dev,
                                      motor_feedback_cb,
                                      NULL,           /* user_data (passed to callback) */
                                      &motor_filter);
    if (filter_id < 0) {
        LOG_ERR("Failed to add CAN filter: %d", filter_id);
        return;
    }

    /* Register state-change callback to detect bus-off automatically */
    can_set_state_change_callback(can_dev, can_state_cb, NULL);

    LOG_INF("CAN subsystem ready. Filter ID: %d", filter_id);
}

/* Monitor for bus-off state — fires when TEC error counter exceeds 255 */
static void can_state_cb(const struct device *dev,
                          enum can_state state,
                          struct can_bus_err_cnt err,
                          void *user_data)
{
    LOG_INF("CAN state=%d TEC=%d REC=%d", state, err.tx_err_cnt, err.rx_err_cnt);

    if (state == CAN_STATE_BUS_OFF) {
        LOG_ERR("CAN bus-off! Re-enabling...");
        /* can_recover re-enables the controller from bus-off state.
         * K_MSEC(100): wait up to 100ms for the controller to complete recovery.
         * After recovery, call can_start() again if needed. */
        can_recover(dev, K_MSEC(100));
    }
}
```

---

### 2.5 UART Async API — Ring Buffer, Idle-Line Callback, NMEA Accumulation

```c
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>
#include <string.h>

LOG_MODULE_REGISTER(gps_uart, LOG_LEVEL_DBG);

/* Ring buffer: 512 bytes holds ~6 full 80-byte NMEA sentences.
 * Size must be a power of 2 for ring_buf_item_declare to work correctly. */
#define UART_RX_BUF_SZ  512
RING_BUF_DECLARE(gps_rb, UART_RX_BUF_SZ);
/* Note: RING_BUF_DECLARE is for byte streams (ring_buf_get/put).
 * RING_BUF_ITEM_DECLARE is for typed items (different API). Use DECLARE for UART. */

static const struct device *gps_uart = DEVICE_DT_GET(DT_NODELABEL(usart1));

/* Semaphore: ISR gives one token when new data arrives.
 * GPS thread takes the token, drains ring buffer, assembles sentences. */
K_SEM_DEFINE(gps_rx_sem, 0, 1);

/* UART interrupt callback: called from ISR context.
 * Rules: no sleeping, no blocking, no heavy computation.
 * One job: move bytes from UART FIFO to ring buffer, give semaphore. */
static void gps_uart_isr(const struct device *dev, void *user_data)
{
    if (!uart_irq_update(dev)) {
        return;  /* not our interrupt, or interrupt not ready yet */
    }

    if (!uart_irq_rx_ready(dev)) {
        return;  /* TX interrupt or other — we only care about RX */
    }

    uint8_t byte;
    int bytes_read = 0;

    /* Drain the hardware FIFO completely.
     * uart_fifo_read returns number of bytes actually read.
     * Loop until FIFO is empty — each call to the ISR may deliver multiple bytes
     * if the CPU was busy and several arrived. */
    while (uart_fifo_read(dev, &byte, 1) == 1) {
        /* ring_buf_put returns number of bytes written.
         * Returns 0 if the ring buffer is full — we lost a byte. */
        if (ring_buf_put(&gps_rb, &byte, 1) == 0) {
            /* This fires if your GPS thread is too slow.
             * Increase UART_RX_BUF_SZ or give gps thread higher priority. */
            LOG_WRN_ONCE("GPS ring buffer full — byte lost");
        }
        bytes_read++;
    }

    if (bytes_read > 0) {
        /* Wake the GPS thread to process what we just received.
         * k_sem_give from ISR is safe. If sem already has a token (thread not yet
         * drained), this call is a no-op (max count=1). No signal is lost because
         * the thread will drain ALL available bytes on next wakeup. */
        k_sem_give(&gps_rx_sem);
    }
}

void gps_uart_init(void)
{
    if (!device_is_ready(gps_uart)) {
        LOG_ERR("GPS UART not ready");
        return;
    }

    /* Register callback, then enable RX interrupts.
     * Order matters: register before enable, otherwise an early interrupt fires
     * with no handler and the byte is dropped. */
    uart_irq_callback_user_data_set(gps_uart, gps_uart_isr, NULL);
    uart_irq_rx_enable(gps_uart);

    LOG_INF("GPS UART initialized");
}

/* GPS processing thread: assembles bytes from ring buffer into complete NMEA lines.
 * Blocks on semaphore — wakes only when ISR says new data arrived.
 * Never busy-polls. Zero CPU waste while waiting for GPS bytes. */
void gps_thread_fn(void *a, void *b, void *c)
{
    gps_uart_init();

    char sentence_buf[128];  /* max NMEA sentence is ~82 chars; 128 is safe margin */
    int buf_idx = 0;

    while (1) {
        /* Block until at least one new byte is in the ring buffer */
        k_sem_take(&gps_rx_sem, K_FOREVER);

        /* Drain ALL bytes currently in ring buffer.
         * Do not read just one byte per semaphore token — the ISR may have
         * delivered 10 bytes before the semaphore woke us. Read until empty. */
        uint8_t byte;
        while (ring_buf_get(&gps_rb, &byte, 1) == 1) {
            if (buf_idx >= (int)sizeof(sentence_buf) - 1) {
                /* Buffer overflow: line longer than expected, or '$' was missed.
                 * Reset and resync on next '$'. */
                LOG_WRN("NMEA buffer overflow — resetting");
                buf_idx = 0;
            }

            sentence_buf[buf_idx++] = (char)byte;

            /* Sentence complete: we have a full line ending in '\n' */
            if (byte == '\n') {
                sentence_buf[buf_idx] = '\0';

                /* Only process sentences we care about.
                 * strncmp checks prefix — handles both $GPRMC and $GNRMC
                 * (GNRMC is the multi-constellation version from newer modules) */
                if (strncmp(sentence_buf, "$GPRMC", 6) == 0 ||
                    strncmp(sentence_buf, "$GNRMC", 6) == 0) {
                    gps_parse_rmc(sentence_buf);
                }

                buf_idx = 0;  /* reset for next sentence */
            }
        }
    }
}

/* Parse $GPRMC sentence and publish to ZBus */
static void gps_parse_rmc(const char *sentence)
{
    char status;
    float lat_raw, lon_raw, speed_kn;
    char ns, ew;

    /* sscanf in Zephyr is available but be aware it uses significant stack.
     * Allocate at least 512 bytes for the GPS thread stack beyond what LOG needs. */
    int n = sscanf(sentence,
                   "$G%*4c,%*6s,%c,%f,%c,%f,%c,%f",
                   &status, &lat_raw, &ns, &lon_raw, &ew, &speed_kn);

    if (n != 6) {
        LOG_DBG("NMEA parse: expected 6 fields, got %d", n);
        return;
    }

    if (status != 'A') {
        /* 'V' = void = no GPS fix. Publish NaN or skip, but do not use raw values.
         * Indoor test: GPS module always returns 'V'. Handle this gracefully. */
        LOG_DBG("GPS no fix (status=%c)", status);
        return;
    }

    /* NMEA coordinate format: DDDMM.MMMM (degrees and decimal minutes, concatenated)
     * 4807.038 means 48 degrees, 07.038 minutes.
     * Converting: degrees + minutes/60.
     * (int)(lat_raw / 100) extracts just the degrees part. */
    float lat_deg = (int)(lat_raw / 100) + fmod(lat_raw, 100.0f) / 60.0f;
    float lon_deg = (int)(lon_raw / 100) + fmod(lon_raw, 100.0f) / 60.0f;
    if (ns == 'S') lat_deg = -lat_deg;
    if (ew == 'W') lon_deg = -lon_deg;

    struct gps_msg msg = {
        .lat  = lat_deg,
        .lon  = lon_deg,
        .speed_mps = speed_kn * 0.514444f,  /* knots to m/s */
    };

    zbus_chan_pub(&gps_chan, &msg, K_NO_WAIT);
}
```

---

### 2.6 ZBus Publish from ISR — Safe Pattern vs Wrong Pattern

```c
#include <zephyr/zbus/zbus.h>

/* === WRONG PATTERN — never do this === */

static void bad_drdy_isr(const struct device *dev,
                          struct gpio_callback *cb, uint32_t pins)
{
    struct imu_sample s;

    /* WRONG 1: i2c_write_read can block (it waits for SCL edges).
     *          Blocking in an ISR hangs the entire system. */
    imu_read_sample(&s);  /* DO NOT CALL I2C FROM ISR */

    /* WRONG 2: zbus_chan_pub with K_FOREVER blocks until the channel is ready.
     *          K_FOREVER in ISR = deadlock if no subscriber is running. */
    zbus_chan_pub(&imu_chan, &s, K_FOREVER);  /* DO NOT USE K_FOREVER IN ISR */
}


/* === CORRECT PATTERN — ISR only signals, thread does work === */

static K_SEM_DEFINE(imu_drdy_sem, 0, 1);  /* 0 initial, 1 max — one pending read */

/* ISR: fires when ICM-42688 INT1 pin goes HIGH (data ready).
 * The only allowed operations in ISR: give semaphore, set a flag, call k_work_submit.
 * Cannot sleep, cannot allocate memory, cannot call LOG (in deferred mode). */
static void imu_drdy_isr(const struct device *dev,
                          struct gpio_callback *cb, uint32_t pins)
{
    /* k_sem_give is ISR-safe. It atomically increments the semaphore count.
     * If the count is already at max (1), this is a no-op — the previous
     * sample was not yet read. This is deliberate: you don't want unlimited
     * backlog from an ISR. The thread will catch up. */
    k_sem_give(&imu_drdy_sem);
}

/* Reading thread: runs at normal thread priority.
 * Waits on semaphore (no CPU waste), then does I2C read, then publishes. */
static void imu_reader_thread(void *a, void *b, void *c)
{
    /* Setup DRDY interrupt */
    static const struct gpio_dt_spec drdy =
        GPIO_DT_SPEC_GET(DT_NODELABEL(imu), int_gpios);
    static struct gpio_callback drdy_cb;

    gpio_pin_configure_dt(&drdy, GPIO_INPUT);
    gpio_init_callback(&drdy_cb, imu_drdy_isr, BIT(drdy.pin));
    gpio_add_callback(drdy.port, &drdy_cb);
    gpio_pin_interrupt_configure_dt(&drdy, GPIO_INT_EDGE_RISING);

    /* Tell IMU to assert INT1 when data is ready (INT_CONFIG register) */
    uint8_t int_cmd[2] = {0x14 /* INT_CONFIG */, 0x02 /* INT1 = data ready */};
    i2c_write(i2c_dev, int_cmd, 2, IMU_ADDR);

    while (1) {
        /* Block here — consumes zero CPU while waiting */
        k_sem_take(&imu_drdy_sem, K_FOREVER);

        struct imu_sample s;
        int rc = imu_read_sample(&s);
        if (rc != 0) {
            LOG_WRN("IMU read failed: %d", rc);
            continue;
        }

        /* K_NO_WAIT: do not block if subscriber is slow.
         * Drop this sample rather than pile up samples in a backlogged queue.
         * IMU runs at 100Hz — a dropped sample is 10ms of data, tolerable. */
        zbus_chan_pub(&imu_chan, &s, K_NO_WAIT);
    }
}
```

---

## PART 3 — Gotcha Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|----------------|-----|
| `i2c_write_read` returns `-ENXIO` on every call | Wrong I2C address — AD0 pin state doesn't match `reg = <0x6X>` in overlay | Use a logic analyzer: watch for address byte sent by master, check if ANY slave ACKs. Or add a bus scan loop that tries addresses 0x60–0x6F | Measure the AD0 pin voltage with a multimeter. Low = 0x68, High = 0x69. Update overlay to match |
| I2C works at 100 kHz but gives corrupted data at 400 kHz | Pull-up resistors too large (10 kΩ or higher): RC rise time exceeds half-period | Logic analyzer: measure rising-edge slope on SDA/SCL. Should reach VCC within 1 µs at 400 kHz. Slow slope = high R | Replace pull-ups with 4.7 kΩ (100–400 kHz) or 2.2 kΩ (1 MHz). Check if breakout has existing pull-ups (parallel reduces R) |
| `i2c_write_read` returns `-EIO` always, even with correct address | I2C bus locked — slave holding SDA LOW after power cycle interrupted a transaction | Logic analyzer: is SDA stuck LOW when SCL is idle HIGH? Measure SDA line with multimeter (should be ~3.3V at idle) | Call `i2c_recover_bus(i2c_dev)`. If this fails, power-cycle the slave. Add recovery call to init code |
| CAN send returns `-ENETDOWN` | `can_start()` was not called before the first send | Add `LOG_INF("CAN state: %d", can_get_state(can_dev))` at startup | Call `can_start(can_dev)` in the init sequence. This is not automatic on STM32 |
| CAN bus seemingly works with two nodes but fails with three or more | Missing or single termination resistor — reflections corrupt signals at higher bus load | Power off the bus. Measure resistance between CAN_H and CAN_L with a multimeter. 60 Ω = correct (two 120 Ω in parallel). 120 Ω = only one terminator | Add 120 Ω resistors at BOTH physical ends of the bus. Not at every node — only the two endpoints |
| CAN bitrate mismatch — no frames received, no errors reported | All nodes on the bus must agree on exact bitrate. Even 1% difference causes bus-off | Use a USB-CAN adapter (CANable) on your laptop with `candump can0`. If `candump` shows nothing when your node transmits, the bus or bitrate is wrong | Set identical `bus-speed` in all device overlays. Verify with oscilloscope: at 1 Mbps, one bit period = 1 µs |
| CAN node goes silent after a few hundred frames | Bus-off state: Transmit Error Counter (TEC) exceeded 255 — node auto-isolated itself | Add `can_set_state_change_callback()`. Log state and TEC/REC counters. Bus-off typically means bitrate mismatch, missing terminator, or CAN_H/CAN_L swapped | Register the state callback and call `can_recover()` on bus-off. Fix the underlying cause (wiring, bitrate) |
| CAN transceiver physically wired but no signal on the bus | CAN transceiver STBY/RS pin asserted (standby mode) | Measure the TX pin of the transceiver with an oscilloscope. Is it toggling? If yes, the transceiver is in standby and blocking output | On SN65HVD230, the RS (slope/standby) pin must be tied LOW (GND) for normal operation. Pull it LOW via 10 kΩ resistor |
| UART receives garbage characters, not ASCII | Baud rate mismatch between host and device | Connect a logic analyzer, decode UART at the suspected rate. Try decoding at 115200 and 9600. The "correct" rate shows ASCII | Set matching `current-speed` in the device overlay AND matching rate in `picocom`/`minicom` |
| NMEA parser occasionally gets partial sentences with garbage | Sentence data split across two ring buffer drains; parsing partial data | Add debug logging inside the ring buffer drain loop to print each received `char`. Watch for `$` and `\n` appearing in the same callback vs different ones | Buffer bytes until `\n`, then parse. Never parse inside the ISR callback |
| GPS ring buffer overflow — bytes lost at 9600 baud | GPS thread priority too low or processing takes too long per tick | Add a byte-loss counter (increment when `ring_buf_put` returns 0). Log it periodically | Increase `UART_RX_BUF_SZ`. Raise GPS thread priority. Move heavy processing (sscanf, math) to after the ring buffer drain loop |
| SDA stuck LOW after power cycle — `i2c_recover_bus` returns non-zero | Slave has partially completed hardware, `i2c_recover_bus` requires bit-bang GPIO support enabled in driver | Try calling recovery immediately at startup (before normal reads). Check if `CONFIG_I2C_STM32_BUS_RECOVERY=y` | Enable `CONFIG_I2C_STM32_BUS_RECOVERY=y` in `prj.conf`. This enables the 9-clock bit-bang routine |
| IMU all-zero sensor readings despite successful WHO_AM_I | Forgot to write PWR_MGMT0 register — sensor is in sleep mode by default | Read back PWR_MGMT0 (0x4E) register — it should be 0x0F for low-noise mode, not 0x00 | Add `i2c_write(i2c_dev, {0x4E, 0x0F}, 2, IMU_ADDR)` in init. Wait 10 ms before first read |
| ISR-based IMU reads hang or reset the system | Calling `i2c_write_read()` from inside an ISR — I2C driver uses semaphores internally | Hard fault or `k_panic` with stack trace pointing into I2C driver | Move I2C read to thread context. ISR must only `k_sem_give()`. Thread does the read |

---

## Quick Reference Card

### I2C Cheatsheet

```
Pull-up values:    4.7 kΩ @ 100–400 kHz   |   2.2 kΩ @ 1 MHz
ICM-42688 addr:    0x68 (AD0=GND)          |   0x69 (AD0=VCC)
Single reg read:   i2c_write_read(dev, addr, &reg, 1, &val, 1)
Burst read N:      i2c_write_read(dev, addr, &start_reg, 1, buf, N)
Bus recovery:      i2c_recover_bus(dev)
Error codes:       -ENXIO = no ACK (wrong addr)  |  -EIO = bus stuck
WHO_AM_I reg:      0x75   Expected value: 0x47
Wake from sleep:   reg 0x4E = 0x0F   Wait 10 ms after
kconfig flag:      CONFIG_I2C=y   CONFIG_I2C_STM32_BUS_RECOVERY=y
```

### CAN Cheatsheet

```
Termination:       120 Ω at EACH end   |   Measure 60 Ω with multimeter (bus off)
Bitrate:           bus-speed = <1000000> in overlay (all nodes must match)
SN65HVD230:        RS pin → GND (standby disable)   |   TJA1050 needs 5V, NOT 3.3V
Start order:       can_start() → can_add_rx_filter() → send/recv
Bus-off recovery:  can_recover(dev, K_MSEC(100))    Callback: can_set_state_change_callback()
STBY check:        RS/STBY pin must be LOW for normal operation
filter mask:       CAN_STD_ID_MASK  (= 0x7FF, all 11 bits must match)
Error codes:       -ENETDOWN = can_start not called  |  state == CAN_STATE_BUS_OFF
```

### UART / GPS Cheatsheet

```
Baud for GPS:      9600 (default most modules)   115200 (configurable)
Ring buffer size:  min 256 bytes; 512 recommended for GPS at 10 Hz
ISR rule:          ONLY ring_buf_put + k_sem_give. Never sscanf, never LOG.
NMEA accumulate:   Buffer until '\n', then parse. One callback ≠ one sentence.
Init order:        uart_irq_callback_set() BEFORE uart_irq_rx_enable()
Status field:      'A' = active (valid fix)   'V' = void (no fix) — skip 'V' data
Coord format:      DDDMM.MMMM → degrees + fmod(raw, 100) / 60
kconfig flags:     CONFIG_UART_INTERRUPT_DRIVEN=y   CONFIG_RING_BUFFER=y
```

### ZBus from ISR

```c
/* ISR: only k_sem_give() */
static void drdy_isr(...) { k_sem_give(&ready_sem); }

/* Thread: does the real work */
while (1) {
    k_sem_take(&ready_sem, K_FOREVER);   /* block, no CPU waste */
    sensor_read(&data);                  /* I2C/SPI OK in thread */
    zbus_chan_pub(&chan, &data, K_NO_WAIT); /* K_NO_WAIT in sensor thread */
}
```

### Endian Conversions for CAN Payloads

```c
/* Big-endian (network order) → native int16 */
int16_t val = (int16_t)sys_get_be16(&frame->data[0]);

/* Native int16 → big-endian in frame */
sys_put_be16((uint16_t)val, &frame->data[0]);
```

### Logic Analyzer Quick Checklist

```
I2C:  Does slave ACK the address byte? (9th bit LOW = ACK)
      Is SDA LOW when SCL is idle? (bus stuck)
      Does the rising edge reach VCC before next SCL edge? (pull-up check)

CAN:  Can you see dominant/recessive transitions? (0V diff / 2V diff)
      Do all nodes agree on bit timing? (decode at your target baud)
      Is the bit rate stable or drifting? (oscilloscope, not logic analyzer)

UART: Decode at suspected baud rate — is it ASCII?
      Watch for '$' then data then '\n' — are they in one callback or split?
```
