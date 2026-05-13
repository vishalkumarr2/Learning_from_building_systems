# Zephyr Devicetree — Describing Your Hardware

## What Is Devicetree?

Devicetree is a **data format that describes hardware** separately from code.
Instead of hardcoding GPIO pin numbers and I2C bus addresses into your C code, you declare them in a `.dts` file, and Zephyr generates C macros you use in code.

**Why this matters**: the same driver code works on any board — only the `.dts` file changes.

---

## The Problem Devicetree Solves

Without devicetree (bad):

```c
/* Hardcoded in source — breaks when you change boards */
#define BUTTON_PIN    5
#define LED_PORT      GPIOC
#define IMU_I2C_ADDR  0x68
#define IMU_BUS       I2C1

gpio_pin_set(GPIOC, 5, 1);  /* what if LED moves to GPIOA pin 7 on new board? */
```

With devicetree (good):

```c
/* Code refers to logical names, never pin numbers */
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
gpio_pin_set_dt(&led, 1);  /* works on any board where "led0" is defined in DTS */
```

---

## Structure of a Devicetree File

Devicetree is a **tree of named nodes**, each with properties:

```dts
/dts-v1/;

/ {                                    /* root node */
    model = "My STM32 Robot Board";

    aliases {
        led0 = &green_led;            /* logical name → node reference */
    };

    leds {                            /* custom node */
        green_led: led_0 {            /* node label: node_name */
            gpios = <&gpioc 7 GPIO_ACTIVE_HIGH>;
            /* third arg = flags: which polarity is "on" */
        };
    };
};

&i2c1 {                               /* modify existing node (from board DTS) */
    status = "okay";                  /* enable this peripheral */
    clock-frequency = <400000>;       /* 400 kHz */

    imu: icm42688@68 {               /* child node: name@address */
        compatible = "invensense,icm42688p";
        reg = <0x68>;
        int-gpios = <&gpiob 3 GPIO_ACTIVE_HIGH>;
    };
};
```

---

## DTS Layers — How They Stack

Real projects use multiple DTS files that overlay each other:

```
Board DTS (e.g., stm32f4_disco.dts)
    └── defines all peripherals, their base addresses, pinmux defaults

Board DTSI (hardware constants, shared between board variants)

SoC DTSI (stm32f4.dtsi, generated from silicon, rarely modified)
    └── defines all hardware blocks: USART1, I2C1, SPI1, etc.

app.overlay  ← YOU write this
    └── enables/configures only what your app needs
```

**You almost always only write `app.overlay`**. It overlays on top of the board DTS, adding or modifying nodes.

---

## The `app.overlay` You Write

```dts
/* app.overlay — placed in your app root directory */

/* Enable I2C1 (inherited from board DTS, marked disabled by default) */
&i2c1 {
    status = "okay";
    clock-frequency = <I2C_BITRATE_FAST>;  /* = 400000 Hz */

    imu: icm42688@68 {
        compatible = "invensense,icm42688p";
        reg = <0x68>;
        int-gpios = <&gpiob 3 GPIO_ACTIVE_HIGH>;
    };
};

/* Enable SPI1 for slave operation */
&spi1 {
    status = "okay";
};

/* Enable CAN1 */
&can1 {
    status = "okay";
    bus-speed = <1000000>;
};

/* Custom aliases for convenience */
/ {
    aliases {
        sensor-i2c = &i2c1;
    };
};
```

---

## Reading Devicetree Values in C

Zephyr generates C macros from the DTS at build time. Headers are in `zephyr/include/zephyr/dt-bindings/` and auto-generated in `build/zephyr/include/generated/`.

### Get a device handle

```c
/* By label */
const struct device *i2c = DEVICE_DT_GET(DT_NODELABEL(i2c1));

/* By alias */
const struct device *led_dev = DEVICE_DT_GET(DT_ALIAS(led0));

/* By node path */
const struct device *spi = DEVICE_DT_GET(DT_PATH(soc, spi_40013000));
```

### Get GPIO spec from DTS

```c
/* In DTS: led_0 { gpios = <&gpioc 7 GPIO_ACTIVE_HIGH>; }; */

static const struct gpio_dt_spec led =
    GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
/* led.port = GPIOC device, led.pin = 7, led.dt_flags = ACTIVE_HIGH */

gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
gpio_pin_toggle_dt(&led);
```

### Get integer property

```c
/* In DTS: uart0 { current-speed = <115200>; }; */

#define BAUD DT_PROP(DT_NODELABEL(uart0), current_speed)
/* Note: property name uses underscore in macro, hyphen in DTS */
```

### Check if a device is available

```c
if (!device_is_ready(i2c)) {
    LOG_ERR("I2C not ready");
    return -ENODEV;
}
```

### Iterate over all children of a node

```c
/* In DTS: leds { led_0 { ... }; led_1 { ... }; }; */

#define INIT_LED(node_id)                                      \
    {                                                          \
        .spec = GPIO_DT_SPEC_GET(node_id, gpios),             \
    },

static const struct {
    struct gpio_dt_spec spec;
} leds[] = {
    DT_FOREACH_CHILD(DT_PATH(leds), INIT_LED)
};
```

---

## Kconfig — prj.conf

Devicetree describes hardware. **Kconfig describes software options**: which drivers to compile, stack sizes, feature flags.

```kconfig
/* prj.conf */

# Enable I2C API
CONFIG_I2C=y

# Enable CAN
CONFIG_CAN=y

# Enable logging
CONFIG_LOG=y
CONFIG_LOG_DEFAULT_LEVEL=3        # 3 = INFO

# Enable ZBus
CONFIG_ZBUS=y

# SPI
CONFIG_SPI=y

# Thread stack sizes (you can also configure per-thread)
CONFIG_MAIN_STACK_SIZE=2048

# Enable stack overflow detection
CONFIG_STACK_SENTINEL=y

# Enable thread analyzer (prints stack usage)
CONFIG_THREAD_ANALYZER=y
CONFIG_THREAD_ANALYZER_USE_PRINTK=y
CONFIG_THREAD_ANALYZER_AUTO=n       # manual: call thread_analyzer_print()

# Shell (useful for debug)
CONFIG_SHELL=y
CONFIG_SHELL_BACKEND_SERIAL=y

# Optional: use 64-bit timestamps
CONFIG_CLOCK_CONTROL=y
```

---

## Bindings — How Zephyr Knows What Properties Are Valid

Every DTS node with `compatible = "vendor,chip"` needs a matching **binding YAML** file that defines its properties.

Zephyr ships thousands of bindings in `zephyr/dts/bindings/`.

```yaml
# Example binding: dts/bindings/sensor/invensense,icm42688p.yaml
description: ICM-42688-P 6-axis IMU

compatible: "invensense,icm42688p"

include: i2c-device.yaml   # inherits reg, status, etc.

properties:
  int-gpios:
    type: phandle-array
    description: Data-ready interrupt output
    required: false
```

If you're writing a **custom driver** and you make a YAML binding file, Zephyr will validate your DTS and generate type-safe macros for your properties.

---

## board vs SoC vs overlay File Roles

| File | Whom | What |
|---|---|---|
| `boards/arm/my_board/my_board.dts` | Board vendor (you for custom boards) | Pin assignments, GPIO numbering, default uart/led nodes |
| `dts/arm/st/stm32f4.dtsi` | Zephyr project | SoC peripherals: base addresses, interrupts |
| `app.overlay` | YOU (app developer) | Enable peripherals, add sensor nodes, configure speed |
| `prj.conf` | YOU (app developer) | Enable/disable software drivers and features |

---

## Common Mistakes

| Mistake | Build error / symptom | Fix |
|---|---|---|
| Forgot `status = "okay"` | No device, `-ENODEV` at runtime | Add `status = "okay"` to node |
| Wrong `compatible` string | Build warning, driver not found | Match exactly to binding YAML |
| Property name with hyphen in C macro | Compile error | Use underscore in C: `current_speed` not `current-speed` |
| `app.overlay` not in app root | Overlay silently ignored | Place next to `CMakeLists.txt` |
| Missing `CONFIG_I2C=y` in prj.conf | Linker error, device null | Add driver Kconfig |
| `device_is_ready` returns false | Init fails silently elsewhere | Add `if (!device_is_ready())` check + log |
