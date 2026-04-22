# Exercises: UART GPS + NMEA Parsing
### Covers: Deep-dive session 08 — ring-buffer accumulation, split-sentence handling, $GNGGA parsing

---

## Section A — Conceptual Questions

**A1.** Explain why you use an interrupt-driven ring buffer (`uart_irq_callback_user_data_set`) instead
of polling `uart_fifo_read()` in a tight loop for GPS reception. What happens to your 100Hz
line-sensor DMA if you poll UART?

<details><summary>Answer</summary>

GPS outputs NMEA sentences at 9600 baud (~960 bytes/sec). Polling with a tight loop means your CPU
is spinning waiting for each byte. At 100Hz the system has 10ms per cycle; a single tight-poll UART
read can consume several milliseconds of that budget waiting for silence between sentences.

With interrupt-driven reception: the UART hardware fires an IRQ when its FIFO has data. The ISR
copies bytes to the ring buffer and returns immediately. The main/GPS thread sleeps on a semaphore
and only wakes when there is data. CPU is free for SPI DMA, ZBus, and scheduling.

If you poll UART in the SPI DMA ISR completion callback (wrong design), you stall the DMA refill
and cause the 100Hz timer to miss its deadline → jitter on the Jetson-side receiver, periodic
timestamp gaps in the ROS2 topic.

</details>

---

**A2.** Describe the **split-sentence** problem with ring buffers. Your ring buffer is 512 bytes.
The typical `$GNGGA` sentence is 82 bytes. Under what specific timing condition will a sentence
appear split across two reads of the ring buffer, and how does your parsing code handle it correctly?

<details><summary>Answer</summary>

**When it happens:** The GPS transmits one byte at a time. Your ring buffer reader wakes on a
semaphore given from the ISR after any number of bytes arrive. If the GPS has sent 60 bytes of a
82-byte sentence and the scheduler wakes your thread, you see an incomplete sentence. The remaining
22 bytes arrive in the next ISR burst.

**Correct handling:** Never call `strtok` or `sscanf` on the ring buffer directly. Instead:

1. Drain all available bytes from the ring buffer into a **line accumulator** (a `static char
   line[256]`).
2. Append each byte to `line`. When you see `\n` (LF), the sentence is complete — parse it and
   reset the accumulator.
3. If `line` fills before seeing `\n` (>128 bytes without newline), the sentence is malformed or
   the ring buffer was overrun — discard and reset.

The key is that the parser is **byte-stateful**: it spans multiple ring-buffer reads using the
static accumulator. The ring buffer just provides byte-level flow; the sentence boundary is
detected by the LF character.

</details>

---

**A3.** Explain the NMEA checksum. Given this sentence:
```
$GNGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*76
```
Describe the algorithm to verify the `*76` checksum without looking it up.

<details><summary>Answer</summary>

XOR all bytes **between** `$` and `*` (exclusive of both). The result is a two-hex-digit value.

```c
uint8_t calc = 0;
const char *p = sentence + 1;   /* skip '$' */
while (*p && *p != '*') {
    calc ^= (uint8_t)*p++;
}
/* p now points at '*', p+1 and p+2 are the hex digits */
uint8_t expected = (uint8_t)strtol(p + 1, NULL, 16);
if (calc != expected) /* checksum fail */;
```

For the sentence above: XOR of `GNGGA,092750.000,...,M,,` = `0x76`. The `*76` matches.

Common mistakes:
- Including the `$` in the XOR → wrong result.
- Including the `*` in the XOR → wrong result.
- Forgetting that the two hex digits are ASCII, not raw bytes.

</details>

---

**A4.** Convert the following raw NMEA coordinate fields to decimal degrees:
- Latitude field: `5321.6802`, hemisphere `N`
- Longitude field: `00630.3372`, hemisphere `W`

Show the formula.

<details><summary>Answer</summary>

NMEA uses `DDMM.MMMM` format (degrees + minutes + fractional minutes):

```
decimal_degrees = DD + MM.MMMM / 60.0
```

**Latitude:**
- DD = 53, MM.MMMM = 21.6802
- 53 + 21.6802 / 60 = 53 + 0.36134 = **53.36134° N**

**Longitude:**
- DD = 6 (leading zero makes it `006`), MM.MMMM = 30.3372
- 6 + 30.3372 / 60 = 6 + 0.5056 = **6.5056°**
- Hemisphere `W` → negate → **-6.5056°**

In C:
```c
double ddmm_to_decimal(double ddmm, char hemi) {
    int deg = (int)(ddmm / 100.0);
    double min = ddmm - (deg * 100.0);
    double dec = deg + min / 60.0;
    if (hemi == 'S' || hemi == 'W') dec = -dec;
    return dec;
}
```

</details>

---

**A5.** Your GPS is configured for 1Hz update rate. Your ZBus GPS channel is being published from
the UART thread at 1Hz, but your SensorFrame packer runs at 100Hz. What GPS data does the
100Hz packer use for the 99 frames between GPS updates? Is this acceptable for an AMR warehouse robot?

<details><summary>Answer</summary>

The packer calls `zbus_chan_read(&gps_chan, &gps, K_NO_WAIT)` which always returns the **last
written value** in the channel. For the 99 frames between 1Hz GPS updates, the packer uses
the most recently decoded GPS fix — i.e., a position that may be up to ~1 second old.

**Acceptability for AMR warehouse robot:**
In a warehouse, a 1Hz GPS update is essentially useless for navigation — GPS doesn't work reliably
indoors. The GPS is typically used only for outdoor last-mile or yard management. Indoors, the
robot uses odometry + IMU (EKF), not GPS.

For the SensorFrame: publishing stale GPS coordinates at 100Hz is fine as long as:
1. The `has_fix` flag is included and correctly reflects signal status.
2. The timestamp in the GPS sub-message reflects the actual fix time, not the pack time.
3. The Jetson receiver ignores GPS fields when `has_fix = false` or `satellites = 0`.

Do NOT simply zero GPS fields when there is no fix — proto3 will silently omit zero fields and the
receiver cannot distinguish "no fix, 0.0°" from "valid fix, precisely on the equator/prime meridian".
Always use a `has_fix` boolean.

</details>

---

## Section B — Practical / Debug Scenarios

**B1.** Your GPS parser works on real hardware but a colleague's unit test fails. The test injects
this sentence via `uart_fifo_read` mock:
```
$GNGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*76\r\n
```
The parser returns `fix_quality = 0` (no fix). The checksum passes. What is the most likely bug?

<details><summary>Answer</summary>

The `fix_quality` field is field index 6 in `$GNGGA` (0-indexed from field 0 = sentence ID). The
bug is almost always an **off-by-one in the `strsep`/`strtok` field counter**.

`$GNGGA` field layout:
```
0: $GNGGA
1: UTC time
2: Latitude
3: N/S
4: Longitude
5: E/W
6: Fix quality  ← this
7: Satellites
8: HDOP
...
```

If the parser uses `strtok(sentence, ",")` starting from the sentence ID (`$GNGGA`) and counts
field 6 as the 7th token (1-indexed), or if it skips the sentence ID and counts field 6 as field
5, it reads the wrong field. `atoi("N")` = 0 → fix_quality = 0.

Secondary cause: the test string uses `\r\n` (Windows line ending). If the parser checks for `\n`
only, the `\r` remains at the end of the last field, and `atoi("\r")` = 0.

</details>

---

**B2.** You inject the shell command `inject_split 40` (splits the sentence after 40 bytes) into your
Zephyr shell for testing. The parser correctly handles the split. Then you inject
`inject_split 1` and the parser hangs. Why?

<details><summary>Answer</summary>

`inject_split 1` delivers one byte per ring-buffer write, giving your thread 82 separate wakeups.
The hang is most likely a **semaphore starvation** issue: the ring-buffer ISR gives the semaphore
82 times faster than the parser thread can take it. If the semaphore is implemented as a binary
semaphore (saturating at count 1) instead of a counting semaphore, many IRQ gives are lost → the
thread reads less data than was written → it blocks on `k_sem_take` with remaining bytes in the
ring buffer that will never trigger another give.

Zephyr's `K_SEM_DEFINE(sem, 0, 1)` — max count of 1 — is the culprit.

Fix: use `K_SEM_DEFINE(uart_data_sem, 0, K_SEM_MAX_LIMIT)` so the count accumulates.

Alternatively: drain the ring buffer completely in one pass (while `ring_buf_get()` returns > 0)
on each semaphore take, rather than reading only one byte per wakeup.

</details>

---

## Section C — Code Reading

**C1.** Find the bug in this NMEA parser:

```c
static bool parse_gngga(const char *sentence, struct gps_data *out) {
    char buf[128];
    strncpy(buf, sentence, sizeof(buf));

    char *tok = strtok(buf, ",");  /* $GNGGA */
    tok = strtok(NULL, ",");       /* time */
    tok = strtok(NULL, ",");       /* lat */
    double lat_raw = atof(tok);
    tok = strtok(NULL, ",");       /* N/S */
    char ns = tok[0];
    tok = strtok(NULL, ",");       /* lon */
    double lon_raw = atof(tok);
    tok = strtok(NULL, ",");       /* E/W */
    char ew = tok[0];
    tok = strtok(NULL, ",");       /* fix quality */
    out->fix_quality = atoi(tok);

    out->latitude  = ddmm_to_decimal(lat_raw, ns);
    out->longitude = ddmm_to_decimal(lon_raw, ew);
    return out->fix_quality > 0;
}
```

<details><summary>Answer</summary>

**Two bugs:**

1. **`strncpy` does not guarantee null-termination.** If `sentence` is exactly 128 bytes or longer,
   `buf` will not be null-terminated. `strtok` will read past the buffer end → undefined behaviour
   (stack corruption or crash). Fix: `buf[sizeof(buf)-1] = '\0';` after the `strncpy`, or use
   `strlcpy`.

2. **No null check on `strtok` return.** If any field is missing (malformed sentence, early `\r\n`,
   truncated ring buffer read), `strtok` returns `NULL`. The next `atof(NULL)` or `tok[0]` is a
   null-pointer dereference → HardFault. Every `tok = strtok(NULL, ",")` must be followed by
   `if (!tok) return false;`.

There is also a style issue: `strtok` is not re-entrant (uses static state). If `parse_gngga` is
called from an ISR while a ZBus listener is mid-parse, the static state is corrupted. Use
`strtok_r` with an explicit `saveptr` for thread-safe parsing.

</details>

---
