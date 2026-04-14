# Zephyr Interrupts and GPIO

## What Is an Interrupt?

An interrupt (IRQ) is a **hardware signal** that tells the CPU "stop what you're doing and handle this now."

When an interrupt fires:
1. CPU finishes current instruction
2. Saves registers to stack  
3. Jumps to **ISR** (Interrupt Service Routine) — your handler function
4. Returns to where it was

Without interrupts, you'd have to constantly poll every peripheral: "any new data from UART? GPIO? Timer?" — wasteful and slow.

With interrupts, peripherals notify you when they need attention.

---

## ISR Rules — The Golden Rules

**ISRs have strict constraints.** They run at the highest priority, interrupting everything else.

```
┌─────────────────────────────────────┐
│       ISR Constraints               │
├─────────────────────────────────────┤
│ ✅ Read/write registers quickly      │
│ ✅ Set a flag / give a semaphore     │
│ ✅ Put a byte in a ring buffer       │
│ ✅ Call k_sem_give(), k_fifo_put()   │
├─────────────────────────────────────┤
│ ❌ Call k_msleep() — NEVER            │
│ ❌ Call k_mutex_lock() — blocks      │
│ ❌ Call printk() / LOG_INF()         │
│ ❌ Allocate memory (malloc)          │
│ ❌ Do heavy computation              │
└─────────────────────────────────────┘
```

**The pattern**: ISR does minimum work (set flag, copy byte), then wakes a thread that does the real work.

```c
/* ISR: tiny */
void button_isr(const struct device *dev,
                struct gpio_callback *cb,
                uint32_t pins)
{
    k_sem_give(&button_sem);   /* wake the handler thread */
}

/* Thread: does real work */
void button_handler_thread(void *a, void *b, void *c)
{
    while (1) {
        k_sem_take(&button_sem, K_FOREVER);
        LOG_INF("Button pressed!");       /* safe here */
        process_button_event();
    }
}
```

---

## GPIO — General Purpose Input/Output

GPIO pins can be:
- **Output**: drive HIGH (VCC) or LOW (GND) — control LEDs, enable/disable sensors, etc.
- **Input**: read HIGH or LOW — button press, sensor indicate pin, etc.
- **Interrupt-capable input**: hardware fires ISR when pin changes

### Devicetree declaration

```dts
/* app.overlay */
/ {
    aliases {
        led0    = &green_led;
        sw0     = &user_button;
    };

    leds {
        green_led: led_0 {
            gpios = <&gpioc 7 GPIO_ACTIVE_HIGH>;
        };
    };

    buttons {
        user_button: button_0 {
            gpios = <&gpioa 0 (GPIO_ACTIVE_LOW | GPIO_PULL_UP)>;
            /* LOW = pressed (typical for buttons), internal pull-up enabled */
        };
    };
};
```

### Output GPIO — LED example

```c
#include <zephyr/drivers/gpio.h>

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

void setup_led(void)
{
    if (!gpio_is_ready_dt(&led)) {
        LOG_ERR("LED GPIO not ready");
        return;
    }

    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
    /* GPIO_OUTPUT_INACTIVE: output mode, starts in "off" state */
    /* GPIO_OUTPUT_ACTIVE: output mode, starts in "on" state */
}

void led_on(void)  { gpio_pin_set_dt(&led, 1); }
void led_off(void) { gpio_pin_set_dt(&led, 0); }
void led_toggle(void) { gpio_pin_toggle_dt(&led); }
```

Note: `GPIO_ACTIVE_HIGH` in DTS means "logical 1 = physical HIGH". `gpio_pin_set_dt(&led, 1)` sets logical 1, which maps to physical HIGH. If the LED is connected active-low (cathode to GPIO), use `GPIO_ACTIVE_LOW` in DTS — then `gpio_pin_set_dt(&led, 1)` sets pin LOW, which turns the LED on. Your code never needs to think about polarity.

### Input GPIO — Button polling

```c
static const struct gpio_dt_spec btn = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);

void setup_button(void)
{
    gpio_pin_configure_dt(&btn, GPIO_INPUT);
    /* The DTS already includes GPIO_PULL_UP in flags */
}

bool is_button_pressed(void)
{
    return gpio_pin_get_dt(&btn) == 1;
    /* Returns logical 1 when button IS pressed */
    /* DTS GPIO_ACTIVE_LOW handles the inversion: physical LOW → logical 1 */
}
```

### Input GPIO — Interrupt on button press

```c
#include <zephyr/drivers/gpio.h>

static const struct gpio_dt_spec btn = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static struct gpio_callback btn_cb_data;
static K_SEM_DEFINE(btn_pressed_sem, 0, 1);

static void btn_isr(const struct device *dev,
                    struct gpio_callback *cb,
                    uint32_t pins)
{
    k_sem_give(&btn_pressed_sem);
}

void button_interrupt_init(void)
{
    if (!gpio_is_ready_dt(&btn)) return;

    gpio_pin_configure_dt(&btn, GPIO_INPUT);

    /* Register callback and enable interrupt */
    gpio_init_callback(&btn_cb_data, btn_isr, BIT(btn.pin));
    gpio_add_callback(btn.port, &btn_cb_data);
    gpio_pin_interrupt_configure_dt(&btn, GPIO_INT_EDGE_TO_ACTIVE);
    /* GPIO_INT_EDGE_TO_ACTIVE = interrupt on edge going to "active" = button pressed */
    /* Other options: GPIO_INT_EDGE_RISING, GPIO_INT_EDGE_FALLING, GPIO_INT_LEVEL_HIGH */
}

void button_thread(void *a, void *b, void *c)
{
    button_interrupt_init();
    while (1) {
        k_sem_take(&btn_pressed_sem, K_FOREVER);
        LOG_INF("Button pressed!");
    }
}
```

---

## Interrupt Priorities on Zephyr ARM

ARM Cortex-M has multiple interrupt priority levels. Lower number = higher priority, runs first.

```
Priority 0  ─── NMI (non-maskable, hardware only)
Priority 1  ─── HardFault
...
Priority 2  ─── IRQ configured by you
Priority 3  ─── IRQ configured by you
...
Priority 15 ─── Lowest priority
```

In Zephyr devicetree:

```dts
&spi1 {
    interrupts = <35 0>;   /* IRQ number 35, priority 0 */
};
```

For GPIO interrupts configured via the GPIO API, Zephyr uses a mid-level priority automatically.

**Critical**: if your ISR calls `k_sem_give()`, that internally calls the scheduler. The ISR priority must be high enough to preempt any thread, but low enough to allow kernel operations. Zephyr handles this via `CONFIG_ZERO_LATENCY_IRQS` and `IRQ_PRIO_LOWEST` when you use the standard GPIO API.

---

## Real Example: IMU Data-Ready Interrupt at 100Hz

This is the recommended pattern for our pipeline — ISR signals a thread which reads and publishes at exactly 100Hz:

```c
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/kernel.h>

/* DTS defines imu node with int-gpios */
static const struct gpio_dt_spec imu_drdy =
    GPIO_DT_SPEC_GET(DT_NODELABEL(imu), int_gpios);

static struct gpio_callback imu_drdy_cb;
static K_SEM_DEFINE(imu_drdy_sem, 0, 1);

static void imu_drdy_isr(const struct device *dev,
                          struct gpio_callback *cb,
                          uint32_t pins)
{
    k_sem_give(&imu_drdy_sem);
}

void imu_thread(void *a, void *b, void *c)
{
    const struct device *i2c = DEVICE_DT_GET(DT_NODELABEL(i2c1));
    uint8_t raw[12];

    /* Wake IMU from sleep */
    uint8_t pwr_buf[2] = { 0x4E, 0x0F };
    i2c_write(i2c, pwr_buf, 2, 0x68);
    k_msleep(10);

    /* Configure IMU INT1 = data-ready pulse */
    uint8_t int_buf[2] = { 0x14, 0x02 };
    i2c_write(i2c, int_buf, 2, 0x68);

    /* Setup GPIO interrupt */
    gpio_pin_configure_dt(&imu_drdy, GPIO_INPUT);
    gpio_init_callback(&imu_drdy_cb, imu_drdy_isr, BIT(imu_drdy.pin));
    gpio_add_callback(imu_drdy.port, &imu_drdy_cb);
    gpio_pin_interrupt_configure_dt(&imu_drdy, GPIO_INT_EDGE_RISING);

    while (1) {
        k_sem_take(&imu_drdy_sem, K_FOREVER);
        /* Data guaranteed ready — IMU pulsed the pin */
        i2c_write_read(i2c, 0x68, (uint8_t[]){0x1F}, 1, raw, 12);
        /* Parse and publish to ZBus */
        publish_imu(raw);
    }
}

K_THREAD_DEFINE(imu_tid, imu_thread, 1024, 5, NULL, NULL, NULL, 0);
```

---

## GPIO vs SPI vs I2C interrupts

Not all interrupts come from GPIOs. Every peripheral has its own interrupt line:

| Interrupt source | How you configure |
|---|---|
| GPIO pin change | `gpio_pin_interrupt_configure_dt()` |
| UART RX data | `uart_irq_rx_enable()` |
| SPI transfer complete | Handled by Zephyr driver internally |
| I2C transfer complete | Handled by Zephyr driver internally |
| Timer overflow | `k_timer_start()` registers callback |
| DMA transfer complete | `dma_config()` with callback |
| CAN frame received | `can_add_rx_filter()` with callback |

For SPI, I2C, UART, CAN — Zephyr's driver handles the hardware interrupts for you. You call high-level `i2c_write_read()` and the driver manages the ISR internally.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Calling `LOG_INF` in ISR | System crash or dropped logs | Never log in ISR, use `k_sem_give` instead |
| Forgetting `gpio_add_callback` | ISR never fires | Both `gpio_init_callback` AND `gpio_add_callback` required |
| Wrong interrupt edge | ISR fires at wrong time | Check datasheet: DRDY high→low or low→high? Use matching `GPIO_INT_EDGE_*` |
| Missing `gpio_pin_configure_dt` | ISR never fires | Must configure pin as input before enabling interrupt |
| GPIO spec references wrong DTS node | Builds but wrong pin | Double-check phandle in DTS |
| Debounce missing for button | Multiple fires per press | Use `k_msleep(50)` after event or apply hardware capacitor |
