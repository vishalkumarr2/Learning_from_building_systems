# 08 — UART GPS: NMEA Parser with Ring Buffer
### STM32H743ZI2 · u-blox NEO-M8N · Zephyr UART IRQ · NMEA 0183

**Status:** 🟡 HARDWARE-GATED  
**Prerequisite:** Session `07` milestone complete  
**Hardware required:** u-blox NEO-M8N GPS module · active antenna (if indoors, use SMA patch)  
**Unlocks:** `09-zbus-nanopb-bridge.md`  
**Time budget:** ~8 hours  
**Mastery plan:** Project 6

---

## Goal of This Session

Receive NMEA sentences from a GPS module via UART, parse `$GNGGA` for lat/lon/alt/fix-status, and log at 1Hz. Handle split-sentence fragments correctly (the key learning objective). Use loopback injection to verify the parser without GPS sky view.

**Milestone**: `lat=35.670000 lon=139.650000 alt=45.2 fix=1` at 1Hz, or with loopback injection: injected NMEA sentences parse correctly including those split across UART callbacks.

---

## Theory: NMEA 0183 Format

```
$GNGGA,123519.00,3567.0000,N,13965.0000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n
│       │         │          │  │          │ │  │   │     │                │
│       │         │          │  │          │ │  │   │     │                checksum (XOR of chars between $ and *)
│       time      latitude   N  longitude  E fix sats hdop altitude
│
GNSS talker (GN=combined, GP=GPS only, GL=GLONASS, GA=Galileo)
```

**GGA fields (0-indexed from sentence start):**
| Index | Field | Notes |
|-------|-------|-------|
| 0 | `$GNGGA` | Message ID |
| 1 | `123519.00` | UTC time hhmmss.ss |
| 2 | `3567.0000` | Latitude ddmm.mmmm |
| 3 | `N` | N/S |
| 4 | `13965.0000` | Longitude dddmm.mmmm |
| 5 | `E` | E/W |
| 6 | `1` | Fix quality: 0=none, 1=GPS, 2=DGPS |
| 7 | `08` | Satellites in use |
| 8 | `0.9` | HDOP |
| 9 | `545.4` | Altitude (MSL) |
| 10 | `M` | Altitude units |

**Latitude conversion:** `ddmm.mmmm` → decimal degrees: `dd + mm.mmmm/60`  
Example: `3567.0000` → `35 + 67.0000/60 = 35 + 1.1167 = 36.1167°N`  
(Note: u-blox outputs `ddmm.mmmm` where `dd` is degrees and `mm.mmmm` is minutes — NOT pure decimal)

**NMEA checksum**: XOR of all ASCII bytes between `$` and `*` (exclusive). Verify on every sentence.

---

## The Key Learning Objective: Split Sentences

UART interrupts fire when the hardware RX FIFO has data — typically 1–64 bytes. A 70-byte NMEA sentence may arrive in 2–4 callbacks:

```
Callback 1:  "$GNGGA,123519.00,3567.0000,N,13"
Callback 2:  "965.0000,E,1,08,0.9,545.4,M,46.9"
Callback 3:  ",M,,*47\r\n"
```

**Wrong approach:** parse inside the IRQ callback. You get a garbage partial sentence.  
**Correct approach:** accumulate bytes into a ring buffer. Processing thread drains the ring buffer looking for `\n` (sentence terminator) before parsing.

---

## Wiring

```
STM32 Nucleo-H743ZI2              NEO-M8N Breakout
────────────────────              ─────────────────
PA9  (USART1_TX) ──────────────── RXD
PA10 (USART1_RX) ──────────────── TXD
3.3V (CN6 pin 4) ──────────────── VCC (3.3V — check module; some require 5V)
GND  (CN6 pin 6) ──────────────── GND
                                   PPS → not connected (for now)
                                   Antenna → SMA active patch (outdoors or near window)
```

**Decoupling:** 100nF capacitor across VCC/GND on the GPS module breakout, placed close to the module pins.

**Default NEO-M8N baud rate:** 9600. NMEA output at 1Hz by default. You can increase to 115200 via UBX-CFG-PRT, but start at 9600.

---

## Zephyr UART IRQ Application

### Devicetree Overlay

```dts
/* boards/nucleo_h743zi2.overlay */
&usart1 {
    status = "okay";
    pinctrl-0 = <&usart1_tx_pa9 &usart1_rx_pa10>;
    pinctrl-names = "default";
    current-speed = <9600>;
};
```

### prj.conf

```kconfig
CONFIG_UART_INTERRUPT_DRIVEN=y
CONFIG_RING_BUFFER=y
CONFIG_LOG=y
CONFIG_MAIN_STACK_SIZE=4096
```

### GPS Parser Implementation

```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LOG_MODULE_REGISTER(gps, LOG_LEVEL_DBG);

#define UART_NODE     DT_NODELABEL(usart1)
#define RING_BUF_SIZE 512

RING_BUF_DECLARE(uart_ring_buf, RING_BUF_SIZE);

/* ── UART IRQ callback ─────────────────────────────────────────── */
static void uart_cb(const struct device *dev, void *user_data) {
    if (!uart_irq_update(dev) || !uart_irq_rx_ready(dev)) {
        return;
    }

    uint8_t byte;
    while (uart_fifo_read(dev, &byte, 1) == 1) {
        if (ring_buf_put(&uart_ring_buf, &byte, 1) == 0) {
            /* Ring buffer full — oldest byte dropped */
            LOG_WRN_ONCE("UART ring buf overflow");
        }
    }
}

/* ── NMEA checksum verifier ────────────────────────────────────── */
static bool nmea_checksum_ok(const char *sentence) {
    /* Sentence: "$GNGGAxxxx...*HH\r\n" */
    const char *p = sentence + 1;   /* skip $ */
    const char *star = strchr(p, '*');
    if (!star) return false;

    uint8_t calc = 0;
    while (p < star) calc ^= (uint8_t)(*p++);

    uint8_t given = (uint8_t)strtol(star + 1, NULL, 16);
    return calc == given;
}

/* ── Parse $GNGGA ──────────────────────────────────────────────── */
typedef struct {
    double  lat_deg;
    double  lon_deg;
    float   alt_m;
    uint8_t fix;
    uint8_t satellites;
    bool    valid;
} gps_data_t;

static gps_data_t parse_gngga(char *sentence) {
    gps_data_t gps = { .valid = false };

    /* Verify checksum first */
    if (!nmea_checksum_ok(sentence)) {
        LOG_WRN("Bad NMEA checksum: %s", sentence);
        return gps;
    }

    /* Tokenise by comma — modifies sentence in place */
    char *fields[15];
    int   n = 0;
    char *token = strtok(sentence, ",");
    while (token && n < 15) {
        fields[n++] = token;
        token = strtok(NULL, ",");
    }
    if (n < 10) return gps;

    /* field[6]: fix quality */
    gps.fix = (uint8_t)atoi(fields[6]);
    if (gps.fix == 0) {
        gps.valid = true;   /* valid parse, no fix */
        return gps;
    }

    /* field[2]: latitude ddmm.mmmm, field[3]: N/S */
    double raw_lat = atof(fields[2]);
    int lat_deg_int = (int)(raw_lat / 100);
    gps.lat_deg = lat_deg_int + (raw_lat - lat_deg_int * 100) / 60.0;
    if (fields[3][0] == 'S') gps.lat_deg = -gps.lat_deg;

    /* field[4]: longitude dddmm.mmmm, field[5]: E/W */
    double raw_lon = atof(fields[4]);
    int lon_deg_int = (int)(raw_lon / 100);
    gps.lon_deg = lon_deg_int + (raw_lon - lon_deg_int * 100) / 60.0;
    if (fields[5][0] == 'W') gps.lon_deg = -gps.lon_deg;

    /* field[9]: altitude */
    gps.alt_m = (float)atof(fields[9]);

    /* field[7]: satellites */
    gps.satellites = (uint8_t)atoi(fields[7]);

    gps.valid = true;
    return gps;
}

/* ── Main processing thread ────────────────────────────────────── */
static char line_buf[128];
static int  line_pos = 0;

void main(void) {
    const struct device *uart_dev = DEVICE_DT_GET(UART_NODE);

    if (!device_is_ready(uart_dev)) {
        LOG_ERR("UART not ready");
        return;
    }

    uart_irq_callback_user_data_set(uart_dev, uart_cb, NULL);
    uart_irq_rx_enable(uart_dev);

    LOG_INF("GPS UART started at 9600 baud, waiting for sentences...");

    while (1) {
        uint8_t byte;
        int got = ring_buf_get(&uart_ring_buf, &byte, 1);

        if (got == 0) {
            k_sleep(K_MSEC(1));
            continue;
        }

        /* Accumulate until '\n' */
        if (byte == '\r') continue;   /* discard CR */

        if (byte == '\n') {
            line_buf[line_pos] = '\0';
            line_pos = 0;

            /* Only process GGA sentences */
            if (strncmp(line_buf, "$GNGGA", 6) == 0 ||
                strncmp(line_buf, "$GPGGA", 6) == 0) {
                gps_data_t gps = parse_gngga(line_buf);
                if (gps.valid) {
                    if (gps.fix > 0) {
                        LOG_INF("lat=%.6f lon=%.6f alt=%.1f fix=%d sats=%d",
                                gps.lat_deg, gps.lon_deg,
                                gps.alt_m, gps.fix, gps.satellites);
                    } else {
                        LOG_INF("No fix yet (satellites visible: %d)", gps.satellites);
                    }
                }
            }
        } else {
            /* Guard against buffer overrun */
            if (line_pos < (int)(sizeof(line_buf) - 1)) {
                line_buf[line_pos++] = byte;
            } else {
                LOG_WRN("Line buffer overflow — discarding");
                line_pos = 0;
            }
        }
    }
}
```

---

## Testing Without GPS Signal: Loopback Injection

Use the STM32 UART shell to inject static NMEA strings and verify the parser without a sky view:

```c
/* Add a shell command to inject test sentences */
#include <zephyr/shell/shell.h>

static int cmd_inject_nmea(const struct shell *sh, size_t argc, char **argv) {
    /* Inject a known-good GGA sentence */
    const char *test_sentence =
        "$GNGGA,123519.00,3540.2000,N,13939.3000,E,1,08,0.9,45.2,M,38.1,M,,*4F\r\n";

    /* Push bytes into the ring buffer as if they came from UART */
    ring_buf_put(&uart_ring_buf, (const uint8_t *)test_sentence, strlen(test_sentence));
    shell_print(sh, "Injected: %s", test_sentence);
    return 0;
}

SHELL_CMD_REGISTER(inject_nmea, NULL, "Inject test NMEA sentence", cmd_inject_nmea);
```

**Test the split-sentence case:**

```c
static int cmd_inject_split(const struct shell *sh, size_t argc, char **argv) {
    /* Same sentence, but injected as two fragments */
    const char *part1 = "$GNGGA,123519.00,3540.2000,N,139";
    const char *part2 = "39.3000,E,1,08,0.9,45.2,M,38.1,M,,*4F\r\n";

    ring_buf_put(&uart_ring_buf, (const uint8_t *)part1, strlen(part1));
    k_sleep(K_MSEC(5));   /* simulate gap between UART callbacks */
    ring_buf_put(&uart_ring_buf, (const uint8_t *)part2, strlen(part2));

    shell_print(sh, "Injected split sentence (part1 + 5ms delay + part2)");
    return 0;
}

SHELL_CMD_REGISTER(inject_split, NULL, "Inject split NMEA sentence", cmd_inject_split);
```

The parser must produce the same result for both `inject_nmea` and `inject_split`. If it fails on `inject_split`, your parser is processing partial callbacks.

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| No output, UART ready | GPS not outputting at 9600 — may be different baud | Connect GPS to host USB-serial adapter, `cat /dev/ttyUSBx` to see raw bytes |
| Garbage characters | Wrong baud rate | Check overlay `current-speed`, GPS module default |
| `Bad NMEA checksum` on every sentence | CR not discarded before checksum calc | Verify `\r` is stripped before XOR |
| Parser produces wrong lat/lon | `ddmm.mmmm` treated as decimal degrees | Use `dd + mm.mmmm/60` conversion |
| "No fix" indefinitely indoors | Active antenna needed, or module cold-start timeout | Move near window; wait 2–5 min for cold fix |
| Ring buffer overflow at 9600 | Processing thread sleeping too long | Reduce sleep to 1ms or use `k_yield()` |

---

## Milestone Checklist

- [ ] `inject_nmea` shell command: `lat=35.670333 lon=139.655000` parsed correctly
- [ ] `inject_split` shell command: same result as `inject_nmea` (split-sentence handled)
- [ ] Checksum failure injected manually: sentence rejected
- [ ] With real GPS outdoors: `fix=1 sats>=4` within 5 minutes of cold start
- [ ] 10-minute continuous run: zero ring buffer overflow warnings
- [ ] Latitude conversion verified: `3540.2000,N` → `35.670333°N` (not `3540.2°`)

---

## Pre-Read for Session 9

Before `09-zbus-nanopb-bridge.md`:
1. Read `zephyr/04_nanopb.md` — proto3 encoding rules, field omission, `max_size` options
2. Read `zephyr/08_zbus.md` — subscriber vs listener, queue depth, priority inversion
3. `exercises/03-zbus-nanopb.md` Q3: proto3 zero-field omission and the variable-length frame problem
4. `00-mastery-plan.md` Project 7 failure points: `pb_encode()` silent false, zero-field omission, ZBus drop at 100Hz

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Setup
- GPS module: NEO-M8N / other: ___
- Default baud rate confirmed: 9600 / other: ___
- Raw NMEA visible on host USB-serial: yes/no

### Parser
- `inject_nmea` result: lat=___ lon=___
- `inject_split` result: lat=___ lon=___ (same as above: yes/no)
- Checksum test: rejected bad sentence: yes/no

### Real GPS
- Fix acquired: yes/no
- Time to first fix: ___min (cold start)
- Satellites at fix: ___

### Issues
- ...
```
