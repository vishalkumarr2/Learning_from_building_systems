# Exercises: SPI Slave + DMA + Double Buffering
### Project 7 — The Hardest Project in the Curriculum
**These exercises assume you have read `04-spi-slave-dma.md` in full.**

---

## Section A — Conceptual Questions

Answer in your own words. If you can explain it simply, you understand it.

---

**A1.** You are debugging an SPI slave implementation. A colleague says: "I'll just arm the DMA inside the CS falling-edge interrupt — that way I only arm it when I actually need it."

Explain specifically why this will fail. Include:
- What is the timing gap between CS assertion and first clock edge?
- What does the SPI hardware do during that gap?
- What is the realistic worst-case latency of a GPIO interrupt on STM32H7?
- What part of every frame will be corrupted, and why that exact portion?

---

**A2.** Describe the D-cache coherency bug in your own words using the "lying librarian" analogy. Then answer:
- Which STM32 families are affected (H7, F4, F7, L4)?
- Why does opening the variable in GDB appear to "fix" the bug temporarily?
- What is the tell-tale byte-count signature that distinguishes this bug from a wiring problem?

---

**A3.** You have two buffers: `buf_A` and `buf_B`. At time T, your packer thread finishes encoding and does a buffer swap. Fill in the blanks:

```
Before swap:
  DMA is reading from: ___
  Packer just wrote into: ___
  
After swap:
  DMA will read from: ___
  Packer will write next frame into: ___
```

Why is this swap called "atomic" on ARM32, and what property of the `tx_active` variable makes a 32-bit write atomic?

---

**A4.** Explain the difference between `SCB_CleanDCache_by_Addr()` and `SCB_InvalidateDCache_by_Addr()`.

For each case below, state which operation to use and why:
1. You just wrote new data into `tx_buf[]` and are about to start a DMA TX transfer
2. DMA RX just completed filling `rx_buf[]` and you want to read the received bytes in C code
3. You are initializing a buffer at startup and want to guarantee it's clean before any DMA access

---

**A5.** The DMA completion callback runs in ISR context. Mark each operation as LEGAL or ILLEGAL in an ISR, and explain why for the two "surprising" ones:

| Operation | Legal / Illegal |
|-----------|----------------|
| `k_sem_give(&frame_done_sem)` | |
| `k_sem_take(&frame_done_sem, K_FOREVER)` | |
| `LOG_INF("frame %u sent", frame_seq)` | |
| `frame_seq++` (global uint32_t) | |
| `k_work_submit(&my_work)` | |
| `spi_transceive(spi_dev, &cfg, &tx, &rx)` | |
| `printk("done\n")` | |
| `arm_spi_dma()` (no-blocking, just sets DMA registers) | |

---

**A6.** Draw (in ASCII art) the timing diagram for a complete 10ms cycle showing:
- The two DMA buffers (which is active, which is being filled)
- When the packer thread runs
- When the Jetson asserts CS and the SPI transfer occurs
- When the buffer swap happens

---

**A7.** Explain the CPOL and CPHA settings:
- What does CPOL control?
- What does CPHA control?
- Mode 0 and Mode 3 are both common. What is different about them?
- If STM32 is configured for Mode 0 and Jetson uses Mode 3, describe exactly what the corrupted data looks like. Is it random, or is there a pattern?

---

**A8.** Why does a sync byte (`0xAA`) alone not fully solve the frame resync problem? Give a specific scenario where the sync byte is present in the payload data and describe what the Jetson receiver would do incorrectly without CRC protection.

---

**A9.** The CRC16 covers only the payload bytes, not the 3-byte header (`0xAA + len`). Argue for why this design is acceptable. Then argue for why including the header in the CRC might be better. Which approach does the code in the study notes use?

---

**A10.** Your packer thread has `CONFIG_MAIN_STACK_SIZE=1024`. The thread encodes a proto using `nanopb`, calls `crc16_ccitt()`, and then calls `sys_cache_data_flush_range()`.

You notice the thread crashes silently once every few hours. What is the most likely cause? How would you diagnose it precisely? What Kconfig values enable stack overflow detection?

---

**A11.** On STM32H7, explain why placing `tx_buf[]` in DTCM (address ~0x20000000) causes a silent DMA failure, while placing it in D2 SRAM (address ~0x30000000) works. What hardware bus limits this?

---

**A12.** A new engineer says: "We don't need double buffering — I'll just make the packer thread write the new frame into the same buffer that DMA just finished sending. After `spi_transceive_async()` completes, DMA is done with that buffer, so it's safe to write, right?"

Is this correct? When would it work? When would it fail? What specific race condition does double-buffering prevent?

---

## Section B — Spot the Bug

Each code snippet has exactly one bug. Identify it, explain why it's a bug, and write the correct code.

---

**B1.** CS interrupt handler:

```c
static void cs_falling_isr(const struct device *dev,
                            struct gpio_callback *cb,
                            uint32_t pins)
{
    /* CS just went low — reload DMA for this transfer */
    tx_spi_buf.buf = tx_buf[tx_active];
    tx_spi_buf.len = tx_len;
    sys_cache_data_flush_range(tx_buf[tx_active], tx_len);
    spi_transceive_async(spi_dev, &slave_cfg, &tx_set, &rx_set,
                         &spi_async_cb, NULL);
}
```

What is the bug? What symptom does it produce?

---

**B2.** The TX-complete callback:

```c
static void spi_async_cb(const struct device *dev, int result, void *userdata)
{
    frame_seq++;
    arm_spi_dma();
    LOG_INF("Frame %u sent, result=%d", frame_seq, result);
    k_sem_give(&frame_done_sem);
}
```

What is the bug? Why does it not crash immediately in testing but may crash in production?

---

**B3.** Buffer definition:

```c
static uint8_t tx_buf[2][300];  /* two 300-byte buffers */
```

And in the packer:

```c
sys_cache_data_flush_range(tx_buf[write_idx], total_len);
```

What is the bug? On what hardware does it matter? What symptom will it produce?

---

**B4.** Buffer swap in packer thread:

```c
/* Swap buffers */
tx_active = write_idx;    /* make new buffer active */
tx_len    = total_len;    /* update length */
```

Why is this order wrong? What race condition can it trigger? Rewrite it correctly.

---

**B5.** The frame builder:

```c
void build_spi_frame(uint8_t *buf, const uint8_t *payload, uint16_t len)
{
    uint8_t local_payload[300];
    memcpy(local_payload, payload, len);

    buf[0] = 0xAA;
    buf[1] = (len >> 8) & 0xFF;
    buf[2] = len & 0xFF;

    /* Encode nanopb into local_payload */
    pb_ostream_t stream = pb_ostream_from_buffer(local_payload, 300);
    pb_encode(&stream, SensorFrame_fields, &sensor_data);

    uint16_t encoded_len = stream.bytes_written;
    memcpy(&buf[3], local_payload, encoded_len);

    uint16_t crc = crc16_ccitt(&buf[3], encoded_len);
    buf[3 + encoded_len]     = (crc >> 8);
    buf[3 + encoded_len + 1] =  crc & 0xFF;

    /* Flush cache before DMA */
    sys_cache_data_flush_range(buf, 3 + encoded_len + 2);

    /* Start DMA */
    spi_transceive_async(spi_dev, &slave_cfg, &tx_set, &rx_set,
                         &spi_async_cb, NULL);
}
```

Hint: think about where `local_payload` lives in memory and what constraints DMA and D-cache have.

---

**B6.** The packer thread's first iteration:

```c
void spi_packer_thread(void *a, void *b, void *c)
{
    while (1) {
        k_sem_take(&frame_done_sem, K_FOREVER);
        int write_idx = 1 - tx_active;

        /* ... build frame into tx_buf[write_idx] ... */
        sys_cache_data_flush_range(tx_buf[write_idx], total_len);
        tx_len    = total_len;
        tx_active = write_idx;
        /* loop back to k_sem_take */
    }
}
```

What happens on the very first iteration? What does `k_sem_take` block on? When (if ever) is `frame_done_sem` given for the first time? Fix it.

---

**B7.** DMA buffer placement:

```c
/* prj.conf */
CONFIG_SPI=y
CONFIG_SPI_STM32_DMA=y
CONFIG_DMA=y

/* spi_slave.c — this is the entire relevant declaration */
static uint8_t __aligned(32) tx_buf[2][256];
```

The system compiles and runs. DMA appears to work for the first transfer, but then becomes unreliable or hangs. What is missing? (Hint: this is an STM32H7 problem.)

---

**B8.** SPI slave configuration:

```c
static const struct spi_config slave_cfg = {
    .frequency = 10000000,    /* 10 MHz */
    .operation = SPI_WORD_SET(8) | SPI_OP_MODE_SLAVE | SPI_TRANSFER_MSB,
};
```

Two bugs. Find both.

---

**B9.** Frame receiver on the Jetson side (Python pseudocode):

```python
def read_frame(spi):
    # Read until sync byte found
    while True:
        b = spi.readbytes(1)[0]
        if b == 0xAA:
            break
    
    # Read length
    header = spi.readbytes(2)
    length = (header[0] << 8) | header[1]
    
    # Read payload + CRC
    data = spi.readbytes(length + 2)
    payload = data[:-2]
    actual_crc = (data[-2] << 8) | data[-1]
    
    # Check CRC
    expected_crc = crc16_ccitt(payload)
    if actual_crc != expected_crc:
        return None  # discard frame

    return payload
```

Is there a logic bug for resync safety? Consider: what happens if `0xAA` appears inside the payload data from the previous frame?

---

**B10.** The async callback arming sequence:

```c
static void spi_async_cb(const struct device *dev, int result, void *userdata)
{
    k_sem_give(&frame_done_sem);   /* signal packer first */
    arm_spi_dma();                 /* then re-arm DMA */
}
```

What timing hazard exists between `k_sem_give()` and `arm_spi_dma()`? Under what condition could the Jetson assert CS and receive garbage because of this ordering? Rewrite with correct order.

---

## Section C — Fill in the Blank

---

**C1.** Complete the buffer declaration. Fill in the three missing pieces and explain each one:

```c
static uint8_t ____________(32) ____________("________")
    tx_buf[2][FRAME_MAX_BYTES];
```

Piece 1 (alignment): `___________` — required because ___________
Piece 2 (attribute): `__attribute__((section("_______")))` — required because ___________
Piece 3 (section name): `____________` — this places the buffer in ___________

---

**C2.** Fill in the Zephyr SPI slave config. All six values needed:

```c
static const struct spi_config slave_cfg = {
    .frequency = ____,        /* why this value for a slave? */
    .operation = SPI_WORD_SET(____) | SPI_OP_MODE_SLAVE | SPI_TRANSFER_MSB
                 | ____ | ____,  /* CPOL and CPHA for Mode 3 */
    .cs = {
        .gpio = { .port = ____ },  /* why this value for the slave side? */
        .delay = 0,
    },
};
```

---

**C3.** Fill in the three cache operations. For each, say the direction (cache→SRAM or SRAM→cache) and when to use it:

```c
// Before DMA TX: force CPU writes to SRAM
___________(tx_buf[active], len);  /* direction: ____ */

// After DMA RX: force CPU to re-read from SRAM
___________(rx_buf, len);          /* direction: ____ */

// Both at once (safest for init):
___________(buf, len);             /* direction: ____ */
```

---

**C4.** Complete the atomic buffer swap. Add the three missing lines:

```c
/* Build frame into write_idx */
int write_idx = 1 - tx_active;
build_frame(tx_buf[write_idx], &sensor_frame, &total_len);

/* Cache flush — fill in: */
___________________________

/* Set length before pointer — fill in: */
___________________________

/* Atomic pointer swap — fill in: */
___________________________
```

---

**C5.** The SPI frame desync recovery has three layers. Complete each layer's description:

| Layer | Mechanism | What it detects/fixes |
|-------|-----------|----------------------|
| Layer 1 | Sync byte `0xAA` | ______________ |
| Layer 2 | ______________ | Detects corrupted/partial frames, forces discard |
| Layer 3 | ______________ | Counts dropped frames, distinguishes transport loss from STM32 fault |

---

**C6.** Complete the packer thread's Kconfig and thread definition. Justify each value:

```c
/* prj.conf additions needed: */
CONFIG_SPI=_
CONFIG_SPI_SLAVE=_
CONFIG_SPI_STM32_DMA=_
CONFIG_DMA=_
CONFIG_CACHE=_           /* needed for sys_cache_data_flush_range on Cortex-M7 */

/* Thread definition: */
K_THREAD_DEFINE(
    spi_packer,      /* name */
    ____,            /* stack size in bytes — why this minimum? */
    spi_packer_thread,
    NULL, NULL, NULL,
    ____,            /* priority — why not 0? why not -1? */
    0,
    0
);
```

---

## Section D — Lab Tasks

Hands-on work. Each task has a specific hardware-verifiable output.

---

**D1. Verify D-cache bug with a logic analyzer**

**Setup:** Implement `spi_slave_push_frame()` but intentionally remove the `sys_cache_data_flush_range()` call.

**Task:**
1. Connect logic analyzer: CS on Ch0, CLK on Ch1, MISO on Ch2
2. Run at 100Hz with a known payload (e.g., rotating counter `0x01, 0x02, 0x03...`)
3. Capture 10 frames
4. Count how many bytes at the start of each frame decode correctly vs. as garbage

**Predicted result (before you run it):** Write your prediction here: "I expect the first ___ bytes to be correct and bytes ___ onwards to be garbage, because ___."

**Verification:** Run it. Compare captured bytes to actual `tx_buf[]` content read via GDB. Add the cache flush back and re-capture to confirm fix.

**Success criteria:** First 32 bytes correct, rest garbage WITHOUT flush. All bytes correct WITH flush. (If your frame is < 32 bytes, create a test frame of >= 64 bytes for this exercise.)

---

**D2. Pre-arm timing measurement**

**Setup:** Implement the WRONG approach (arm DMA in CS ISR) and the CORRECT approach (arm in TX-complete callback).

**Task:**
1. Add a GPIO toggle at the START of `arm_spi_dma()` and another at the END
2. Add a second GPIO toggle in the CS falling ISR at entry
3. Capture on logic analyzer: CS-falling Ch0, Arm-start Ch1, Arm-end Ch2
4. Measure time from CS-falling to Arm-start vs. time from CS-falling to first SPI clock

**With WRONG approach — record:**
- CS-falling to Arm-start: ___ µs
- First SPI clock arrival: ___ ns (from Jetson config)
- Number of bytes that are 0xFF before DMA is ready: ___

**With CORRECT approach — record:**
- Time from previous TX-complete to CS-falling: ___ ms
- During that window, DMA is pre-armed for: ___ ms
- First byte correct: Y/N

**Success criteria:** With correct approach, Arm-end GPIO falls BEFORE CS-falling GPIO. With wrong approach, Arm-end falls AFTER first SPI clock.

---

**D3. Implement CRC16 desync recovery**

**Task:** On the Jetson side (Python or C++), implement a SPI reader that:
1. Scans for sync byte `0xAA`
2. Reads 2-byte length
3. Reads `length + 2` bytes (payload + CRC)
4. Validates CRC
5. If CRC fails: seeks the NEXT `0xAA` in the already-read bytes, tries again
6. Logs a desync event counter

**Test:**
- Inject a bad frame manually (flip one byte in the payload before sending)
- Verify the Jetson reader detects the CRC failure
- Verify recovery: the NEXT correct frame is received and its CRC validates

**Success criteria:** `crc_failures` counter increments exactly once per injected bad frame. The following frame is received correctly. The system does not hang or require a restart.

---

**D4. Sequence number monitoring**

**Task:** Add a `uint32_t frame_seq` counter to the STM32 frame header (bytes 3–6 as big-endian uint32). On the Jetson, monitor the sequence:
1. Parse `frame_seq` from every received frame
2. Track `expected_seq = last_seq + 1`
3. If `actual_seq != expected_seq`: log `drop_count += (actual_seq - expected_seq)`

**Test:**
- Normal operation: verify `drop_count` stays at 0 for 10 seconds @ 100Hz (= 1000 frames)
- Introduce a forced delay in the packer thread (`k_msleep(100)` for one iteration)
- Verify Jetson's `drop_count` increments by exactly the number of skipped frames

**Success criteria:** Zero drops under normal operation confirmed over 1000 consecutive frames. Forced drop correctly detected and counted.

---

**D5. Stack sizing audit for the packer thread**

**Task:**
1. Enable `CONFIG_THREAD_STACK_INFO=y` and `CONFIG_STACK_SENTINEL=y`
2. Set packer stack to 512 bytes (intentionally too small)
3. Run for 30 seconds at 100Hz with nanopb encoding enabled
4. Add a shell command `spi stack` that prints:
   ```
   Packer stack used: %u / %u bytes (%.0f%% utilized)
   ```
5. Resize the stack to have exactly 25% headroom above the measured high-water mark

**How to get stack usage:**
```c
size_t unused;
k_thread_stack_space_get(&spi_packer, &unused);
printk("Used: %u bytes", CONFIG_PACKER_STACK_SIZE - unused);
```

**Success criteria:** 
- With 512-byte stack: `CONFIG_STACK_SENTINEL=y` triggers a kernel panic (expected — confirms stack is too small)
- With correct stack: `k_thread_stack_space_get()` reports <75% utilization after 30s
- Shell command output matches: any value ≤75% is a pass

---

## Section E — Timing Calculations

Show your work.

---

**E1.** SPI transfer time budget

The Jetson drives the SPI bus at **10 MHz**. Each clock cycle transfers 1 bit.

Given a frame of **150 bytes**:
1. How many clock cycles are required to transfer the full frame?
2. How long does the transfer take in microseconds?
3. The SPI transfer occupies the MISO line for how many µs out of each 10ms slot (at 100Hz)?
4. What percentage of the 10ms budget is the line "busy"?

---

**E2.** CPU budget comparison

At 100Hz with 150-byte frames, CPU polling vs DMA:

| Metric | CPU polling | DMA |
|--------|------------|-----|
| Bytes transferred | 150 | 150 |
| Transfer duration | `___ µs` | `___ µs` |
| CPU cycles occupied (STM32H7 @ 480MHz) | `___ cycles` (100% CPU during transfer) | `___ cycles` (only ISR: assume 1µs total) |
| CPU cycles per second @ 100Hz | `___` | `___` |
| CPU budget consumed (% of 480M cycles/sec) | `___` % | `___` % |

Fill in all blanks and explain: at what `bytes × Hz` product does DMA become mandatory? (Hint: use ">5% CPU budget" as your threshold.)

---

**E3.** Pre-arm timing margin calculation

Scenario: Jetson spidev configuration has `delay_usecs = 5` (5µs between CS assertion and first clock edge).

Your ISR environment:
- Maximum IRQ masking duration (critical sections): 8 µs
- Other ISRs that may preempt (UART RX): up to 3 µs
- `arm_spi_dma()` execution time: 0.8 µs

**Questions:**
1. What is the total maximum time from CS falling to when your ISR-based arming finishes? 
2. How does this compare to the 5µs margin the Jetson provides?
3. How many bytes of the frame will be corrupted if the ISR takes the full maximum duration?
4. At a `delay_usecs = 0` (production setting), is ISR-based arming safe under any circumstances?

---

**E4.** Cache line math

Cortex-M7 D-cache: 16KB, 4-way set-associative, 32-byte cache lines.

Your frame buffer is 267 bytes.

1. How many cache lines does a 267-byte buffer span? (Ceiling division: round up)
2. `SCB_CleanDCache_by_Addr` works on cache-line granularity. If your buffer starts at byte offset 4 within a cache line (not aligned), how many extra bytes at the start and end does the flush cover?
3. Those "extra" bytes belong to the adjacent struct. If you flush them, what could go wrong?
4. You declare `uint8_t buf[267]`. The compiler places it at address `0x30000010`. Is this 32-byte aligned? Show your calculation.
5. You declare `uint8_t __aligned(32) buf[267]`. The compiler guarantees address `0x30000020`. How many cache lines are flushed by `SCB_CleanDCache_by_Addr(buf, 267)`?

---

**E5.** Throughput and frequency headroom

Your packer thread takes the following time per frame (worst case):
- `get_latest_sensor_frame()`: 50 µs
- `pb_encode()`: 80 µs
- `build_frame()` (header + CRC): 5 µs
- `sys_cache_data_flush_range()`: 3 µs
- Thread wakeup/scheduling overhead: 20 µs

Total worst-case packer time: **158 µs**

1. What is the maximum frame rate (Hz) the packer can sustain?
2. At 100Hz, what is the packer's duty cycle (% of time spent running)?
3. You want 40% headroom for `pb_encode()` to get slower when the message grows. What is the maximum allowed encode time budget?
4. If `pb_encode()` exceeds budget for 3 consecutive frames, the watchdog should trigger. In the packer thread, how would you detect this condition using `k_uptime_get_32()`?

---

## Answer Hints (Check After Attempting)

These are not full answers — just enough to validate your reasoning.

**A2:** Only M7-based STM32s affected. GDB reads memory, which causes a cache fill from SRAM, showing what the cache SHOULD contain — but DMA already ran. Fingerprint: 32 bytes correct = one cache line.

**A3:** Before: DMA reads from tx_buf[tx_active], packer writes into tx_buf[1-tx_active]. After swap: flipped. Atomicity: ARM32 STR to a 32-bit aligned address is a single indivisible instruction.

**A8:** If `0xAA` appears at byte 73 of the payload, and the Jetson misses one real sync byte, it will see the `0xAA` at byte 73 and start reading from there. The wrong length bytes it reads next will give a nonsense length, and it attempts to process garbage.

**B1:** Bug: arming DMA inside CS falling ISR. By the time the ISR body runs and `spi_transceive_async()` completes its internal setup (~1–10µs), the SPI clock is already running. First 1–4 bytes sent as 0xFF.

**B2:** Bug: `LOG_INF()` called from ISR context. LOG uses a deferred work queue (allocation, potential blocking). Will likely not crash immediately if log buffer has space, but can corrupt the log subsystem or cause a hard fault under load.

**B3:** Bug: no `__aligned(32)`. `sys_cache_data_flush_range` rounds to cache line boundaries, potentially covering adjacent memory and corrupting it.

**B4:** Swap `tx_active` before `tx_len`. ISR reads new buffer index but old length → sends wrong byte count.

**B5:** Bug: `local_payload` is a stack variable. Stack is in DTCM (0x20000000 on H7). DMA cannot access DTCM via DMA1/2. Also: cache flush on `buf` is correct, but the encode went into `local_payload` (stack) and was memcpy'd into `buf` — this copy works, but it's wasteful. The actual DMA bug is the temporary encode target.

**B9:** Bug: after finding `0xAA`, if the sync byte was in the middle of the previous frame's payload, the `length` read is the next 2 bytes of payload data (garbage). A better design: when CRC fails, seek the NEXT `0xAA` in the already-received bytes from this "frame" before issuing another SPI transfer.

**B10:** Bug: `k_sem_give()` before `arm_spi_dma()`. The packer thread may wake, check that packer just gave sem, write to the idle buffer, and call `arm_spi_dma()` in the thread context — racing with the ISR's `arm_spi_dma()`. Call `arm_spi_dma()` first to ensure DMA is re-armed before the packer can do anything.

**E1:** 150 bytes × 8 bits = 1200 bits / 10,000,000 Hz = **120 µs**. That's 1.2% of 10ms.

**E3:** Max ISR latency: 8 + 3 + 0.8 = **11.8µs**. Margin: 5µs. ISR-based arming exceeds the margin by 6.8µs → **not safe**. Bytes corrupted at 10MHz: 11.8µs / 0.1µs per bit × 1bit/8bits = ~14 bytes. At `delay_usecs=0`: **never safe** — even best-case IRQ latency (25ns response + 800ns arm) may exceed 0ns margin.

**E4:** 267 bytes / 32 bytes per line = 8.34 → **9 cache lines**. If start offset = 4, first flush covers 28 bytes before (belonging to prior struct) and potentially 12 bytes after (next struct). This can corrupt adjacent memory.
