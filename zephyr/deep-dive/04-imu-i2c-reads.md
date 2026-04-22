# 04 — IMU I2C Reads: ICM-42688-P
### STM32H743ZI2 · Zephyr I2C API · ICM-42688-P IMU

**Status:** 🟡 HARDWARE-GATED — fill this in when ICM-42688-P breakout arrives  
**Prerequisite:** `03-dma-cache-gotchas.md` milestone complete  
**Hardware required:** Nucleo-H743ZI2 · ICM-42688-P breakout (SparkFun or Adafruit) · 4× 2.2kΩ resistors  
**Unlocks:** `05-100hz-spi-bridge.md`  
**Time budget:** ~4 hours

---

## Goal of This Session

Read accelerometer and gyroscope data from the ICM-42688-P over I2C at 100Hz.
Log values to Zephyr shell. Verify against hand-shaking the board (gyro Z should
spike when you rotate, accel X/Y/Z should show ≈9.81 m/s² when stationary).

**Milestone**: 100Hz I2C reads, no `EIO` errors, values physically plausible.

---

## Pre-Wiring: Know Your Breakout Board

The ICM-42688-P has TWO power pins:
- `VDD`: core power (1.71V – 3.6V)
- `VDDIO`: I/O interface voltage (1.71V – 3.6V)

Most breakout boards (SparkFun, Adafruit) have an onboard regulator and level shifter — they
accept 3.3V on both `VDD` and `VDDIO`. **Verify this on your specific breakout schematic.**

If your breakout does NOT have a level shifter and your IMU uses 1.8V VDDIO, driving it
with 3.3V STM32 signals will damage the IMU slowly (latch-up at elevated temperature).

### Checking Your Breakout

Look up the schematic of your specific breakout board (the part number is on the silkscreen).
Things to confirm:
- [ ] Input voltage range includes 3.3V
- [ ] I2C pull-ups present on the breakout (value?)
- [ ] CS pin pulled high (needed if using I2C mode, since ICM-42688 defaults to SPI if CS=0)

---

## Theory: I2C Register Read Pattern

The ICM-42688 uses a standard I2C register-read sequence:

```
START → ADDR+W → REG_ADDR → REPEATED_START → ADDR+R → DATA[0] → ... → DATA[n-1] → STOP

Zephyr API: i2c_write_read(dev, addr, &reg, 1, buf, len)
                                        ↑                ↑
                                  register address    receive buffer
```

### Key Registers (I2C address 0x68 when AD0=GND, 0x69 when AD0=VCC)

| Register | Address | Content |
|----------|---------|---------|
| `WHO_AM_I` | 0x75 | Should return 0x47 — use this to verify connection |
| `ACCEL_DATA_X1` | 0x1F | Accel X high byte (16-bit big-endian) |
| `ACCEL_DATA_X0` | 0x20 | Accel X low byte |
| `GYRO_DATA_X1` | 0x25 | Gyro X high byte |
| `GYRO_DATA_X0` | 0x26 | Gyro X low byte |
| `PWR_MGMT0` | 0x4E | Power mode — write 0x0F to enable accel+gyro at full power |
| `ACCEL_CONFIG0` | 0x50 | Accel range: 0x01=±2g (highest sensitivity) |
| `GYRO_CONFIG0` | 0x4F | Gyro range: 0x06=±500°/s |

For continuous 100Hz reads, configure the ODR (output data rate) register too:
- `ACCEL_CONFIG0[3:0] = 0b0111` = 200Hz ODR (decimate with software to 100Hz)
- `GYRO_CONFIG0[3:0] = 0b0111` = 200Hz ODR

Full register map: ICM-42688-P datasheet, Appendix A.

---

## Step-by-Step

### Step 1: Wiring

```
ICM-42688 Breakout    STM32 Nucleo-H743ZI2
────────────────────  ─────────────────────
VIN / 3V3             3.3V (CN6 pin 4)
GND                   GND  (CN6 pin 6)
SDA                   I2C1_SDA = PB9 (CN7 pin 4)
SCL                   I2C1_SCL = PB8 (CN7 pin 2)
AD0                   GND  → I2C address 0x68
CS                    3.3V → select I2C mode (CS=HIGH)

I2C pull-ups:
If breakout has pull-ups: no extra resistors needed
If not: add 2.2kΩ from SDA→3.3V and SCL→3.3V
```

**Saleae Logic 8:**
```
Ch5 → I2C1_SDA (PB9)
Ch6 → I2C1_SCL (PB8)
```
Add I2C analyzer in Logic 2: address=7-bit, speed=400kHz

### Step 2: Verify I2C Bus with Stuck-Bus Recovery

Before writing application code, verify the bus works and add stuck-bus recovery:

```c
#include <zephyr/drivers/i2c.h>

#define ICM42688_ADDR  0x68
#define REG_WHO_AM_I   0x75
#define EXPECTED_WHOAMI 0x47

static const struct device *i2c_dev = DEVICE_DT_GET(DT_NODELABEL(i2c1));

int verify_imu_connection(void) {
    if (!device_is_ready(i2c_dev)) {
        LOG_ERR("I2C device not ready");
        return -ENODEV;
    }

    uint8_t who_am_i;
    uint8_t reg = REG_WHO_AM_I;
    int ret = i2c_write_read(i2c_dev, ICM42688_ADDR, &reg, 1, &who_am_i, 1);
    if (ret < 0) {
        LOG_ERR("I2C read failed: %d (is bus stuck? check wiring)", ret);
        return ret;
    }

    if (who_am_i != EXPECTED_WHOAMI) {
        LOG_ERR("WHO_AM_I = 0x%02x, expected 0x47", who_am_i);
        return -EIO;
    }

    LOG_INF("ICM-42688-P found, WHO_AM_I = 0x47 ✓");
    return 0;
}
```

### Step 3: Configure IMU Registers

```c
static int configure_icm42688(void) {
    /* Write helper: reg addr + value in one i2c_write call */
    #define IMU_WRITE(reg, val) do { \
        uint8_t buf[] = { (reg), (val) }; \
        int r = i2c_write(i2c_dev, buf, 2, ICM42688_ADDR); \
        if (r) return r; \
    } while (0)

    /* Enable accel + gyro at low-noise mode */
    IMU_WRITE(0x4E, 0x0F);      /* PWR_MGMT0: accel+gyro ON, LN mode */
    k_sleep(K_MSEC(50));        /* Startup time: 50ms for LN mode */

    /* Accel: ±2g, 200Hz ODR */
    IMU_WRITE(0x50, 0x17);      /* ACCEL_CONFIG0: AFS=±2g, AODR=200Hz */

    /* Gyro: ±500°/s, 200Hz ODR */
    IMU_WRITE(0x4F, 0x67);      /* GYRO_CONFIG0: GFS=±500dps, GODR=200Hz */

    LOG_INF("ICM-42688-P configured: ±2g accel, ±500dps gyro, 200Hz ODR");
    return 0;
}
```

### Step 4: Read Sensor Data at 100Hz

```c
typedef struct {
    float accel_x, accel_y, accel_z;  /* m/s² */
    float gyro_x, gyro_y, gyro_z;     /* rad/s */
} imu_data_t;

/* Sensitivity from ICM-42688 datasheet:
   ±2g   → 16384 LSB/g
   ±500°/s → 65.5 LSB/(°/s) → 3753.5 LSB/(rad/s) */
#define ACCEL_SCALE  (9.81f / 16384.0f)
#define GYRO_SCALE   (1.0f / 3753.5f)

static int read_imu(imu_data_t *out) {
    /* Read 12 bytes starting at ACCEL_DATA_X1 (0x1F):
       ACCEL_X_H, ACCEL_X_L, ACCEL_Y_H, ACCEL_Y_L, ACCEL_Z_H, ACCEL_Z_L,
       GYRO_X_H,  GYRO_X_L,  GYRO_Y_H,  GYRO_Y_L,  GYRO_Z_H,  GYRO_Z_L  */
    uint8_t reg = 0x1F;
    uint8_t raw[12];
    int ret = i2c_write_read(i2c_dev, ICM42688_ADDR, &reg, 1, raw, 12);
    if (ret < 0) return ret;

    int16_t ax = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t ay = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t az = (int16_t)((raw[4] << 8) | raw[5]);
    int16_t gx = (int16_t)((raw[6] << 8) | raw[7]);
    int16_t gy = (int16_t)((raw[8] << 8) | raw[9]);
    int16_t gz = (int16_t)((raw[10] << 8) | raw[11]);

    out->accel_x = ax * ACCEL_SCALE;
    out->accel_y = ay * ACCEL_SCALE;
    out->accel_z = az * ACCEL_SCALE;
    out->gyro_x  = gx * GYRO_SCALE;
    out->gyro_y  = gy * GYRO_SCALE;
    out->gyro_z  = gz * GYRO_SCALE;
    return 0;
}

void main(void) {
    verify_imu_connection();
    configure_icm42688();

    imu_data_t data;
    while (1) {
        if (read_imu(&data) == 0) {
            LOG_INF("A: x=%+.3f y=%+.3f z=%+.3f g; G: x=%+.3f y=%+.3f z=%+.3f r/s",
                    data.accel_x, data.accel_y, data.accel_z,
                    data.gyro_x, data.gyro_y, data.gyro_z);
        }
        k_sleep(K_MSEC(10));   /* 100Hz */
    }
}
```

### Step 5: Physical Sanity Check

Hold the board flat on a table. Expected readings:
```
A: x≈0.00  y≈0.00  z≈9.81 g;   G: x≈0.00  y≈0.00  z≈0.00 r/s
```

Rotate the board 90° around X axis. Expected:
```
A: x≈0.00  y≈9.81  z≈0.00 g;   G: x≈(spike during rotation)
```

Shake the board rapidly:
```
A: (varies, large values during shake)
G: (varies, large values during rotation components)
```

---

## Stuck I2C Bus Recovery

If you power-cycle while an I2C transaction was in progress, SDA may be held low.
`i2c_write_read` will return `-EIO` forever. Fix at init time:

```c
/* In devicetree overlay: use GPIO to bit-bang 9 clock pulses */
static void recover_i2c_bus(void) {
    const struct gpio_dt_spec scl = GPIO_DT_SPEC_GET(DT_NODELABEL(i2c1_scl_gpio), gpios);
    const struct gpio_dt_spec sda = GPIO_DT_SPEC_GET(DT_NODELABEL(i2c1_sda_gpio), gpios);

    gpio_pin_configure_dt(&scl, GPIO_OUTPUT_HIGH);
    gpio_pin_configure_dt(&sda, GPIO_INPUT);

    /* Clock out up to 9 pulses to release stuck slave */
    for (int i = 0; i < 9; i++) {
        if (gpio_pin_get_dt(&sda)) break;   /* SDA released = slave freed */
        gpio_pin_set_dt(&scl, 0);
        k_busy_wait(5);                     /* 100kHz half-period */
        gpio_pin_set_dt(&scl, 1);
        k_busy_wait(5);
    }

    /* Generate STOP condition */
    gpio_pin_configure_dt(&sda, GPIO_OUTPUT_LOW);
    k_busy_wait(5);
    gpio_pin_set_dt(&scl, 1);
    k_busy_wait(5);
    gpio_pin_set_dt(&sda, 1);

    /* Re-initialize I2C peripheral */
    i2c_recover_bus(i2c_dev);   /* Zephyr 3.5+ has this directly */
}
```

---

## Milestone Checklist

- [ ] `WHO_AM_I` reads 0x47 — I2C connection confirmed
- [ ] Logic analyzer decodes WHO_AM_I read correctly
- [ ] Stationary: accel_z ≈ 9.81 m/s², accel_x,y ≈ 0
- [ ] Board rotation: accel axis swaps as expected
- [ ] 100Hz read loop runs for 60 seconds without `-EIO`
- [ ] Stuck-bus recovery tested: power-cycle mid-transaction, recovery code restores operation

---

## Pre-Read for Session 5

Before `05-100hz-spi-bridge.md`:
1. Re-read `zephyr/07_jetson_ros2_bridge.md` — the full bridge architecture
2. Re-read `zephyr/08_spi_protocol.md` — frame format, protobuf encoding
3. `00-mastery-plan.md` "Wrong Timestamps (3 days lost)" — plan for `clock_gettime` on Jetson side

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Breakout Board Used
- Model: ...
- Has onboard pull-ups: yes/no (value if yes: ___)
- Has level shifter: yes/no

### Connection Verification
- WHO_AM_I result: 0x___

### Calibration
- Stationary accel_z: ___ m/s² (expected ≈9.81)
- Gyro bias (stationary): gx=___ gy=___ gz=___ rad/s

### Stuck Bus Test
- Triggered artificially: yes/no
- Recovery worked: yes/no
- Time to recovery: ___ms

### ODR Measured
- Expected 200Hz, actual: ___Hz (measure with Logic 2 trigger count / time)

### Surprises
- ...
```
