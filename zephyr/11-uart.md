# UART — Universal Asynchronous Receiver-Transmitter

## What Is UART?

UART is the **simplest serial protocol** — just two wires, no clock, no address, point-to-point only.

"Asynchronous" means there's no shared clock signal. Both sides agree on speed beforehand (the baud rate) and each side has its own clock. The protocol includes start/stop bits to signal when bytes begin and end.

Used for: GPS modules, Bluetooth modules (HC-05, HM-10), WiFi modules (ESP8266), debug consoles, legacy sensors, LoRa modules, GNSS receivers — anything that sends ASCII text or simple binary frames at moderate speeds and doesn't need to be fast.

In our robot: UART is how the STM32 receives **GPS/GNSS data** ($GPRMC sentences at 10Hz) and outputs **debug logs** to a laptop.

---

## Physical Wires — Just 2

```
STM32 TX ──────────────────────► GPS RX
STM32 RX ◄────────────────────── GPS TX
GND ────────────────────────────── GND
```

TX/RX are **crossed**: STM32's TX goes to the other device's RX, and vice versa.

No pull-up resistors needed (unlike I2C). Voltage is actively driven HIGH or LOW.

**RS-232** (older standard): ±12V swings. Needs level converter MAX232.  
**TTL serial** (modern embedded): 0V / 3.3V or 0V / 5V. Direct connection between boards.

---

## Baud Rate — Speed Contract

Baud = symbols per second. For UART, 1 baud = 1 bit per second.

Both sender and receiver must be configured to the same baud rate. If mismatched, you get garbage.

Common baud rates:

| Baud | Bytes/sec (approx) | Use |
|---|---|---|
| 9600 | 960 | Legacy, slow sensors |
| 115200 | 11520 | Debug console ← most common |
| 230400 | 23040 | Faster debug |
| 921600 | 92160 | High-speed binary |
| 1000000 | 100000 | Custom high-speed link |

At 115200 baud, a 80-byte GPS sentence takes ~7ms. At 10Hz GPS, that's 7% bus utilization — fine.

---

## Frame Format

Each byte is wrapped in start/stop bits:

```
Idle:  1 1 1 1 1 1 1 1 1 1 1 1...
       ↑
       Line is HIGH at idle

One byte:
       ┌──────────────────────────────────────┐
 Line: │S│ D0│ D1│ D2│ D3│ D4│ D5│ D6│ D7│ P│ STOP │
       └──────────────────────────────────────┘
        ↑                                   ↑  ↑
        Start bit (LOW)                  Parity Stop bit(s) → HIGH
```

- **Start bit**: line goes LOW → receiver starts its internal bit timer
- **Data bits**: 5-9 bits, typically 8 (one byte)
- **Parity bit**: optional error detection (None/Odd/Even)
- **Stop bits**: 1 or 2 HIGH bits → separator between bytes

"9600 8N1" means: 9600 baud, 8 data bits, No parity, 1 stop bit. The most common default.

---

## UART in Zephyr

### Devicetree

```dts
/* app.overlay */
&usart1 {
    status = "okay";
    pinctrl-0 = <&usart1_default>;
    pinctrl-names = "default";
    current-speed = <9600>;   /* GPS modules often default to 9600 */
};
```

### Polling API (simple, OK for low rate)

```c
#include <zephyr/drivers/uart.h>

static const struct device *gps_uart = DEVICE_DT_GET(DT_NODELABEL(usart1));

/* Send a string */
void uart_send_str(const char *s)
{
    while (*s) {
        uart_poll_out(gps_uart, *s++);
    }
}

/* Read one byte (blocks until available) */
int uart_recv_byte(uint8_t *out)
{
    return uart_poll_in(gps_uart, out);
    /* returns 0 if byte ready, -1 if no data yet */
}
```

**Warning**: `uart_poll_in` is non-blocking (returns -1 if nothing ready). You'd need a loop to actually block. For real use, use the interrupt-driven API below.

### Interrupt-Driven + Ring Buffer (production approach)

```c
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>

#define RING_BUF_SIZE 512
RING_BUF_ITEM_DECLARE(uart_rb, RING_BUF_SIZE);

static const struct device *gps_uart = DEVICE_DT_GET(DT_NODELABEL(usart1));
static K_SEM_DEFINE(uart_rx_sem, 0, 1);

/* Called from ISR when RX data arrives */
static void uart_rx_isr(const struct device *dev, void *user_data)
{
    if (!uart_irq_update(dev)) return;

    if (uart_irq_rx_ready(dev)) {
        uint8_t byte;

        while (uart_fifo_read(dev, &byte, 1) == 1) {
            if (!ring_buf_put(&uart_rb, &byte, 1)) {
                LOG_WRN("UART ring buffer full");
            }
        }
        k_sem_give(&uart_rx_sem);  /* wake reader thread */
    }
}

void uart_init(void)
{
    uart_irq_callback_set(gps_uart, uart_rx_isr);
    uart_irq_rx_enable(gps_uart);
}

/* Read one complete NMEA line (blocks until '\n') */
int uart_read_line(char *buf, size_t max_len)
{
    size_t idx = 0;

    while (idx < max_len - 1) {
        k_sem_take(&uart_rx_sem, K_FOREVER);

        uint8_t byte;
        while (ring_buf_get(&uart_rb, &byte, 1) == 1) {
            buf[idx++] = byte;
            if (byte == '\n') {
                buf[idx] = '\0';
                return idx;
            }
        }
    }
    return -ENOSPC;
}
```

### Parsing GPS NMEA Sentences

GPS modules output ASCII sentences at 1Hz or 10Hz:

```
$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
        ↑       ↑  ↑         ↑           ↑         ↑
       time  status lat      lon        speed    date
```

```c
void gps_thread_fn(void *a, void *b, void *c)
{
    char line[128];

    uart_init();

    while (1) {
        uart_read_line(line, sizeof(line));

        /* Only care about $GPRMC (recommended minimum) */
        if (strncmp(line, "$GPRMC", 6) != 0) continue;

        /* Simple parser — production code would use minmea library */
        float lat, lon, speed_kn;
        char ns, ew, date[7], status;

        /* sscanf is OK for NMEA strings under Zephyr */
        int n = sscanf(line, "$GPRMC,%*6s,%c,%f,%c,%f,%c,%f",
                       &status, &lat, &ns, &lon, &ew, &speed_kn);

        if (n == 6 && status == 'A') {  /* A = active/valid fix */
            /* Convert DDDMM.MMMMM → decimal degrees */
            float lat_deg = (int)(lat / 100) + fmod(lat, 100) / 60.0f;
            float lon_deg = (int)(lon / 100) + fmod(lon, 100) / 60.0f;

            if (ns == 'S') lat_deg = -lat_deg;
            if (ew == 'W') lon_deg = -lon_deg;

            struct gps_msg msg = {
                .lat_deg  = lat_deg,
                .lon_deg  = lon_deg,
                .speed_mps = speed_kn * 0.514444f,
            };
            zbus_chan_pub(&gps_chan, &msg, K_NO_WAIT);
        }
    }
}
```

### Using UART for Debug Console

```c
/* prj.conf */
/* CONFIG_UART_CONSOLE=y      (default — uses UART for printk) */
/* CONFIG_LOG=y               */
/* CONFIG_LOG_BACKEND_UART=y  */

/* Your code */
#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER(main, LOG_LEVEL_DBG);

void main(void) {
    LOG_INF("System started");          /* → UART0 at 115200: "main: System started" */
    LOG_DBG("Sensor value: %d", 42);
    LOG_WRN("Buffer nearly full");
    LOG_ERR("Init failed: %d", -EIO);
}
```

Plug in a USB-UART adapter (CP2102, CH340) to see logs on laptop with `minicom` or `picocom`:

```bash
picocom -b 115200 /dev/ttyUSB0
```

---

## UART vs I2C vs SPI vs CAN

| | UART | I2C | SPI | CAN |
|---|---|---|---|---|
| Wires | 2 | 2 | 4+ | 2 (differential) |
| Max speed | ~10 Mbps | 1 MHz | 50+ MHz | 8 Mbps (FD) |
| Multi-slave | No | Yes (address) | Yes (CS per slave) | Yes (arbitration) |
| Cable length | ~15m | <1m | <1m | ~25m |
| Clock | None (async) | Master | Master | None (async, NRZ) |
| Error detection | Optional parity | ACK/NACK | None | CRC+ACK |
| Best for | GPS, BT, debug | Onboard sensors | Fast IMU | Long cables, motors |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| TX→TX, RX→RX (not crossed) | Nothing received | Cross TX↔RX |
| Baud rate mismatch | Garbage characters | Set same baud on both sides |
| Blocking in ISR | System freezes | ISR just puts byte in ring buffer |
| Reading before `uart_irq_rx_enable` | No interrupts fire | Call irq enable at startup |
| No ring buffer → byte overrun at 115200 | Data loss | Use ring buffer, never poll from thread loop |
| Missing common GND | Works nearby, fails far | Always connect GND |
