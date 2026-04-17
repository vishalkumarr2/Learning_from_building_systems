# SPI Slave + DMA + Double Buffering — Study Notes
### Project 7: 100Hz STM32 SPI Slave to Jetson
**Hardware:** STM32 Nucleo-H743ZI2 · Zephyr RTOS · Jetson Orin · Logic Analyzer (essential)

> **Electronics prerequisite:** If shift registers, CPOL/CPHA modes, or CS timing feel unfamiliar, read first:
> - SPI physics → `electronics/05-spi-deep-dive.md` (shift register ring, 4 modes, signal integrity)
> - DMA buffers need D-cache awareness → `electronics/01-passive-components.md` won't help here, but understanding memory-mapped I/O is essential

---

## PART 1 — ELI15 Concept Explanations

---

### 1.1 SPI master vs slave — the clock owner determines everything

**The core idea: one side is the boss, one side waits.**

In any SPI conversation, exactly one side controls the clock. That side is the **master**. The other side is the **slave** — it cannot initiate a transfer, cannot decide when to talk, cannot do anything until the master starts clocking.

In our system:
- **Jetson = master.** It owns the clock. It decides *when* to read sensor data.
- **STM32 = slave.** It can never initiate. It must have data ready and waiting *before* the Jetson asks.

**Why this is hard:** When you are the master, you are in control. You call `spi_transceive()` when *you* are ready. When you are the slave, you have zero control over timing. The Jetson can assert CS and start clocking at any moment — 5ms from now, 10ms from now, 11.3ms from now. You don't know. You just have to always be ready.

```
Jetson (master) decides:          STM32 (slave) must:
  "I want sensor data now"     →   Already have the frame in the DMA buffer
  CS goes LOW                  →   DMA begins sending immediately (no delay!)
  Clock starts (10 MHz)        →   SPI hardware shifts bits out one per clock
  CS goes HIGH                 →   Transfer done. Load new frame before next CS.
```

**The bus when it's happening at 10 MHz:**

```
CS:    ────┐                                  ┌────  (LOW = active)
           └──────────────────────────────────┘
CLK:           ┌┐ ┌┐ ┌┐ ┌┐ ┌┐ ┌┐ ┌┐ ┌┐ ┌┐ ┌┐
               └┘ └┘ └┘ └┘ └┘ └┘ └┘ └┘ └┘ └┘
MISO:   ↑STM32 puts data on this line (1 bit per clock)
         ^─ data must be valid BEFORE first clock edge ─^
```

Each clock cycle transfers 1 bit. At 10 MHz, 1 byte takes 800 ns. A 150-byte frame takes 120 µs.

**Mental model for the rest of this chapter:** You (the STM32) are a waiter at a restaurant. The Jetson is your customer. You don't know when they'll order. If they snap their fingers (assert CS) and you don't already have their plate ready (DMA armed), they get nothing. You cannot start cooking after they snap — it's too late.

---

### 1.2 The CS timing problem — why 100ns is not enough time to react

**The brutal timing reality.**

When the Jetson asserts CS (pulls it LOW), the STM32's SPI hardware immediately starts looking for clock edges. Between CS going low and the first clock edge, there may be as little as 100 nanoseconds to 1 microsecond, depending on the Jetson's spidev configuration.

Meanwhile, an interrupt takes:
- ~1–10 µs to fire, depending on what else is running
- Plus context save/restore time
- Plus your ISR code execution time
- Plus DMA arm time

**The math for why you cannot arm DMA in the CS interrupt:**

```
Timeline with WRONG approach (arm DMA in CS ISR):
  t=0ns:    CS goes LOW
  t=100ns:  First clock edge arrives (Jetson starts clocking!)
  t=0ns to t=1500ns: SPI hardware shifts first 12 bits → 0xFF garbage
  t=1500ns: Your ISR finally fires
  t=2000ns: Your ISR calls spi_dma_reload()
  t=2500ns: DMA armed
  Result: first 1-2 bytes of EVERY frame are garbage
```

**The correct approach:** The DMA must already be armed and pointing at a valid buffer **before** CS goes low. This means you arm it in the *previous transfer's completion callback*, not in the CS interrupt.

```
Timeline with CORRECT approach (arm DMA on previous TC):
  t=-5ms:   Previous transfer completes → DMA immediately re-armed
  t=0ns:    CS goes LOW
  t=100ns:  First clock edge arrives
  t=100ns:  SPI hardware reads from DMA buffer → sends real data
  Result: all bytes correct, every time
```

This is the #1 bug that trips everyone up on SPI slave. You will lose 2 days to this bug if you don't know about it in advance.

---

### 1.3 What DMA is — the "post office" analogy

**The problem without DMA:**

Imagine you need to mail 1,000 letters. The naive approach: you personally walk each letter to the mailbox, one at a time. You insert letter 1. You wait for the mailbox to be empty. You insert letter 2. You wait. You insert letter 3...

This is exactly what the CPU does when sending data without DMA:

```c
// CPU manually moves every byte to the SPI hardware register
for (int i = 0; i < 256; i++) {
    SPI1->DR = buffer[i];           // write byte to SPI hardware
    while (!(SPI1->SR & SPI_SR_TXE)); // BUSY-WAIT until SPI is ready
}
// CPU is 100% occupied the entire time — doing nothing useful
```

At 10 MHz, 256 bytes takes 205 µs. During those 205 µs, the CPU cannot run your sensor thread, cannot update state, cannot do anything. At 100 Hz, this burns 2% of your CPU budget on mailbox trips.

**The post office analogy for DMA:**

DMA is a dedicated post office employee. Instead of you walking each letter to the mailbox:
1. **You write an address on one envelope** (configure DMA registers: source, destination, count)
2. **You hand it to the post office** (enable DMA: write CR |= DMA_SxCR_EN)
3. **The post office handles ALL delivery** (DMA reads bytes from SRAM, writes to SPI_DR, one per clock)
4. **The post office rings your doorbell when done** (DMA fires an interrupt when all N bytes transferred)
5. **You go do other things** until the doorbell rings

```
WITHOUT DMA (CPU does all work):
  CPU: moves byte 0 → waits → byte 1 → waits → byte 2 → waits → ... (200µs blocked)

WITH DMA (CPU delegates):
  CPU: "DMA, copy buffer[0..255] → SPI_DR. Interrupt me when done." (takes ~200ns)
  CPU: runs sensor fusion, updates state, services other threads
  DMA: quietly trickles bytes to SPI at wire speed (no CPU help needed)
  DMA: fires interrupt
  CPU: handles completion (~500ns): load next buffer, signal packer thread
```

**Physical picture of what happens:**

```
┌─────────────────────┐  AHB bus  ┌──────────────┐  APB bus  ┌────────────────┐
│  SRAM (tx_buf[])    │ ─────────►│  DMA engine  │ ─────────►│  SPI1->DR reg  │
│  (your data here)   │           │  (hardware)  │           │  (shift out)   │──► MISO pin
└─────────────────────┘           └──────────────┘           └────────────────┘
         CPU's involvement: ~200ns to set up. Zero bytes moved by CPU.
```

The DMA engine has its own bus master capability — it can read SRAM and write peripheral registers independently, without asking the CPU permission for each byte.

---

### 1.4 DMA vs CPU transfer comparison — why DMA is necessary at 100 Hz

**The numbers argument:**

At 100 Hz, you transfer one frame every 10 ms.

| Metric | CPU polling | DMA |
|--------|------------|-----|
| Frame size | 150 bytes | 150 bytes |
| Transfer time (10MHz SPI) | 120 µs actively occupied | 120 µs (DMA working, CPU free) |
| CPU cycles consumed | ~34,000 (at 480MHz: 120µs × 480) | ~1,500 (setup + completion ISR) |
| CPU budget burned at 100Hz | ~3.4M cycles/sec = **0.7% of 480MHz** per transfer, continuously | ~150K cycles/sec = 0.03% |
| Can CPU run other threads? | No (spinning in loop) | Yes |

**The real cost is not bytes, it's setup/teardown:**

Even if a 150-byte DMA transfer "only" costs 0.7% CPU, the *mode* matters enormously. A CPU spinning in a tight polling loop:
1. **Locks out all other threads** at the same priority (no preemption during the loop)
2. **Generates worst-case ISR latency** (other ISRs queue up waiting for the CPU)
3. **Cannot sleep**, so the CPU is at full power draw the entire transfer

With DMA:
1. CPU sets up DMA registers (~200 ns)
2. CPU does `k_sem_take()` or just runs other threads
3. DMA completion fires an interrupt (~500 ns ISR)
4. Total CPU involvement: ~700 ns per 10 ms frame = 0.007%

At 100 Hz, DMA isn't optional — it's the only way to have a functional system where other subsystems also get CPU time.

---

### 1.5 Double buffering — the "two-lane highway" analogy

**The problem a single buffer creates:**

Suppose DMA is currently sending `buf[0..149]` over SPI. Your packer thread finishes encoding the next frame and writes new data into `buf[0..149]`. What just happened?

You wrote to a buffer that DMA is actively reading. DMA will now send a mix of old and new bytes — partial corruption, every time. This is a race condition.

**The two-lane highway:**

Imagine a highway with two lanes. Cars (data) always drive in lane A. Workers (your packer thread) always work on lane B. Every 10 ms, there's an instant lane swap: A becomes the work lane, B becomes the driving lane. Traffic never stops; workers never have to dodge cars.

```
Two buffers: A and B
"Active buffer":  DMA is reading this and sending it → NEVER write here
"Idle buffer":    CPU/packer is writing the next frame here → DMA never reads this

State at t=0ms:
  Buffer A: [frame 1 data] ←── DMA is sending this
  Buffer B: [empty]        ←── packer is building frame 2

State at t=1ms:
  Buffer A: [frame 1 data] ←── DMA is sending this
  Buffer B: [frame 2 data] ←── packer finished building frame 2

State at t=1ms (SWAP happens atomically — one word write):
  Buffer B: [frame 2 data] ←── DMA will now send this on next CS
  Buffer A: [frame 1 data] ←── packer will now write frame 3 here

No moment where DMA reads while packer writes: they always operate on different buffers.
```

**The swap must be atomic.** On ARM32, writing a 32-bit aligned word is atomic (single instruction, cannot be interrupted mid-write). This is why `tx_active` is an `int` (32-bit) — the pointer swap `tx_active = write;` is one MOV instruction.

**Visual timeline at 100 Hz:**

```
     0ms      1ms      2ms      5ms (CS edge)  10ms     11ms     15ms (CS edge)
     │        │        │        │              │        │        │
DMA: ╠═══buff A═══════════════╣ ╠════buff B════════════════════╣
PKR: ╠═══ building B ═╣       │ ╠══════ building A ═══╣       │
     │                │        │                        │        │
     SWAP happens here ────────┘                        SWAP ───┘

PKR never touches buff A when DMA is reading buff A. ✓
DMA never reads buff B when PKR is writing buff B. ✓
```

---

### 1.6 The pre-arming problem — why you must arm DMA BEFORE CS goes low

This is the most important timing rule in this entire project. It deserves its own section.

**The interrupts-are-too-slow argument:**

```
CS → first clock: 100ns to 1µs (depends on Jetson spidev delay_usecs setting)
Interrupt latency on STM32H7:
  - NVIC response time: ~12 CPU cycles at 480MHz = 25ns (fine!)
  - BUT: IRQ may be masked during critical sections: up to 10µs
  - BUT: Other ISRs may be running: up to 10µs for each
  - Realistic worst-case from CS-falling to your ISR code: 1–10µs

Reality: your CS ISR fires AFTER the first 1–4 bytes have already been clocked out.
```

**Why it feels like it should work (and why it doesn't):**

In testing, you might see it work "mostly." The Jetson's default spidev settings often add 10-100µs between CS and first clock (to allow for setup time). Your ISR fires in time. Tests pass.

Then you reduce the Jetson's SPI delay to production settings (0µs), and the first 2 bytes of every frame become `0xFF`. You've been lucky, not correct.

**The correct mental model:**

DMA must be armed and ready at all times, not reactively when CS fires. Think of it like a gun safety: the gun is always loaded and has a round chambered. When you pull the trigger (CS goes low), firing is instantaneous.

```
WRONG: arm on CS falling edge
  CS fires → ISR runs → DMA armed → already missed first bytes

CORRECT: arm on previous transfer complete
  Previous TX done → ISR: re-arm DMA with new buffer immediately
  ...time passes (5ms, 10ms, 3ms — doesn't matter)...
  CS fires → DMA already armed → first bit sent correctly
```

**Code implication:**

```c
// In the TX complete callback (CORRECT):
void spi_tx_complete_cb(void)
{
    // Immediately re-arm for next transfer
    arm_dma_for_next_frame();    // ← arm NOW, before CS fires
    // Signal packer thread that the just-sent buffer is now free
    k_sem_give(&buffer_free_sem);
}

// In the CS ISR (WRONG: too late):
void cs_falling_isr(void)
{
    arm_dma_for_next_frame();    // ← WAY too late, clock already running
}
```

---

### 1.7 STM32H7 D-cache coherency — the "lying librarian" analogy

**This bug will make you feel like you're going insane.** It is responsible for 2+ lost debugging days for most engineers who encounter it. Read this section twice.

**What D-cache is:**

The STM32H7 has a Cortex-M7 core with a 16 KB L1 data cache (D-cache). This cache sits between the CPU and SRAM. When you write to a variable:

1. The write goes into the cache (fast — 1 clock)
2. The cache will write it back to SRAM later (eventually, when the cache line is evicted)

When you read a variable that's in the cache, you get the cached value (fast). If it's not in cache, you fetch from SRAM (slower, loads into cache).

**The librarian analogy:**

Imagine a library where you ask the librarian for a book. The librarian has a small office (the cache) where they keep copies of frequently requested books. When you ask for a book, the librarian gives you their cached copy instantly.

Now someone else (DMA) sneaks into the library stacks and **updates the original book**. But the librarian didn't notice — they still have the old copy in their office. Next time you ask for that book, the librarian confidently hands you the **old version** from their office.

```
The bug:
  1. CPU writes new sensor data to tx_buf[]    → goes into D-cache (SRAM not updated!)
  2. You call start_dma_tx()                   → DMA reads directly from SRAM
  3. SRAM still has old/stale data             → DMA sends old data over SPI!
  4. Jetson receives garbage                   → but no error flag anywhere
  5. You open GDB and read tx_buf[]            → GDB forces cache fill → shows CORRECT data!
```

**Why GDB makes it worse:** GDB reads variables by forcing the CPU to load from memory. This fills the cache with the correct data and shows you the right value. But the DMA already ran from the stale SRAM. GDB lies to you by fixing the problem in the act of showing it.

**The tell-tale diagnostic sign:**

If you have a 150-byte frame and the D-cache line size is 32 bytes:
- First ~32 bytes correct: the first cache line happened to be evicted to SRAM before DMA ran
- Bytes 32–149: garbage: those cache lines had NOT been written back to SRAM

This "first N bytes correct, rest is garbage" pattern WHERE N ≈ 32 (one cache line) is the D-cache coherency fingerprint.

**The fix — SCB_CleanDCache_by_Addr:**

```c
// Before starting DMA TX:
SCB_CleanDCache_by_Addr((uint32_t *)tx_buf[write_idx], frame_len);
// "Clean" = force CPU cache lines → SRAM (write-back)
// Now SRAM has the fresh data that DMA will read
start_dma_tx(tx_buf[write_idx], frame_len);
```

**For DMA RX (receiving data INTO a buffer):**

```c
start_dma_rx(rx_buf, rx_len);
// ... wait for DMA complete ...
// DMA wrote to SRAM, but CPU cache might have stale data at those addresses
SCB_InvalidateDCache_by_Addr((uint32_t *)rx_buf, rx_len);
// "Invalidate" = mark cache lines as stale → force CPU to re-read from SRAM
uint8_t first_byte = rx_buf[0];  // NOW reads fresh SRAM data
```

**The M0/M3/M4 exception:** These cores (including STM32F103, F303, F401/F411) have NO D-cache. Cache coherency is only a problem on Cortex-M7 (H7, F7) and Cortex-A (Jetson). If you switch to an F4 in testing and the cache bug disappears, that's why — not because your fix worked.

---

### 1.8 SCB_Clean vs SCB_Invalidate — "clean" and "invalidate" are NOT the same

People get these backwards. The confusion causes the wrong operation to be applied and the bug persists.

**Clean = write dirty cache lines BACK to SRAM (push)**

Think: "clean up the cache by pushing its contents to memory."

- Use before **DMA reads** from a buffer you just wrote
- Direction: cache → SRAM (writing data OUT of cache into SRAM)
- After Clean: SRAM has fresh data. Cache still has the data too (not evicted).

```c
// I wrote new data into tx_buf. Now DMA will read from SRAM.
// Clean pushes my cache writes to SRAM so DMA sees fresh data.
SCB_CleanDCache_by_Addr((uint32_t *)tx_buf, count);
// DMA TX safe to start
```

**Invalidate = mark cache lines as stale, force CPU re-read from SRAM (pull)**

Think: "invalidate the cache's knowledge, forcing it to go back to source."

- Use after **DMA writes** into a buffer that the CPU will then read
- Direction: SRAM → cache (forcing CPU to re-fetch from SRAM next access)
- After Invalidate: cache has no valid copy of those addresses. Next CPU access fetches from SRAM.

```c
// DMA RX just filled rx_buf in SRAM. CPU cache may have stale data.
// Invalidate forces CPU to re-read from SRAM on next access.
SCB_InvalidateDCache_by_Addr((uint32_t *)rx_buf, count);
uint8_t val = rx_buf[0];  // CPU re-fetches from SRAM — correct!
```

**CleanInvalidate = both at once (safest, slightly slower):**

```c
// Use when you're not sure which direction is the problem, or for initialization
SCB_CleanInvalidateDCache_by_Addr((uint32_t *)buf, count);
```

**The alignment constraint (catches people off guard):**

These functions operate on cache lines, which are 32 bytes on Cortex-M7. If your buffer doesn't start on a 32-byte aligned address, the function may clean/invalidate the wrong range — potentially corrupting adjacent data.

**This is why buffers must be `__aligned(32)`:**

```c
// CORRECT: 32-byte aligned → cache line operations work correctly
static uint8_t __aligned(32) tx_buf[2][300];

// WRONG: compiler places at arbitrary alignment → cache ops corrupt neighboring data
static uint8_t tx_buf[2][300];
```

---

### 1.9 SPI modes — the CPOL/CPHA matrix, explained with timing diagrams

Four SPI modes exist. Each device specifies which mode it wants. If your controller and device disagree, every byte reads as garbage — with no error message.

**Two independent settings:**

- **CPOL** (Clock **Pol**arity): What is the clock doing when idle (between transfers)?
  - CPOL=0: clock **idle LOW** (sits at 0 between transfers)
  - CPOL=1: clock **idle HIGH** (sits at 1 between transfers)

- **CPHA** (Clock **Pha**se): When do you sample the data — on the first edge or the second?
  - CPHA=0: sample on the **first** edge after CS goes low
  - CPHA=1: sample on the **second** edge after CS goes low

**All four modes — ASCII timing diagrams:**

```
Mode 0 (CPOL=0, CPHA=0) — clock idle LOW, sample on RISING edge:
  CS:   ──┐                                              ┌──
          └──────────────────────────────────────────────┘
  CLK:        ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
              │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
         ─────┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └─────
  MOSI: ──[D7]───[D6]──[D5]──[D4]──[D3]──[D2]──[D1]──[D0]──
                ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
                SAMPLE on RISING edge (first edge of each bit)
  Use for: ICM-42688 IMU, W25Q flash, SD cards, most sensors

Mode 1 (CPOL=0, CPHA=1) — clock idle LOW, sample on FALLING edge:
  CLK:        ┌──┐  ┌──┐  ┌──┐  ┌──┐
         ─────┘  └──┘  └──┘  └──┘  └─────
  MOSI: ──────[D7]──[D6]──[D5]──[D4]──
                   ↑     ↑     ↑     ↑
              SAMPLE on FALLING edge (second edge of each bit)
  Use for: some ADCs, AT45DB flash

Mode 2 (CPOL=1, CPHA=0) — clock idle HIGH, sample on FALLING edge:
  CLK:   ──────┐  ┌──┐  ┌──┐  ┌──┐  ┌──────
               └──┘  └──┘  └──┘  └──┘
  MOSI: ──[D7]──[D6]──[D5]──[D4]──[D3]──
              ↑     ↑     ↑     ↑     ↑
         SAMPLE on FALLING edge (first edge when CPOL=1)
  Use for: some DACs, MPU-9250 in CPOL=1 mode

Mode 3 (CPOL=1, CPHA=1) — clock idle HIGH, sample on RISING edge:
  CLK:   ──────┐  ┌──┐  ┌──┐  ┌──┐  ┌──────
               └──┘  └──┘  └──┘  └──┘
  MOSI: ───────[D7]──[D6]──[D5]──[D4]──
                    ↑     ↑     ↑     ↑
                SAMPLE on RISING edge (second edge when CPOL=1)
  Use for: ICM-42688 also supports this; some EEPROMs
```

**The practical checker before starting any SPI project:**
1. Open the device datasheet
2. Find the table labeled "SPI Mode" or "CPOL/CPHA"
3. Set your controller's `operation` field to match

In Zephyr:
```c
// Mode 0:
.operation = SPI_WORD_SET(8) | SPI_OP_MODE_SLAVE   // default CPOL=0, CPHA=0

// Mode 3:
.operation = SPI_WORD_SET(8) | SPI_OP_MODE_SLAVE | SPI_MODE_CPOL | SPI_MODE_CPHA
```

---

### 1.10 CRC16 framing — why a magic header alone isn't enough

**The "false start byte" problem:**

Suppose your frame starts with sync byte `0xAA`. Your Jetson reads bytes until it sees `0xAA`, then reads the next N bytes as a frame. Simple and fast.

What if byte 73 of your sensor payload happens to be `0xAA`? The Jetson is in sync now, reads normally. But one packet later, it misses the real `0xAA` (e.g., due to a brief power glitch). Now it finds the `0xAA` inside the previous frame's payload. It starts reading from the *middle* of a frame, treating everything as a new frame. Your nav stack is now running on corrupted data.

**What CRC adds:**

CRC (Cyclic Redundancy Check) is a mathematical checksum. The sender runs all payload bytes through a polynomial calculation and appends a 2-byte result. The receiver runs the same calculation and compares. If they disagree, the frame is discarded.

```
Frame layout:
  [0xAA] [len_hi] [len_lo] [.... payload bytes ....] [crc_hi] [crc_lo]
     1       1        1         len bytes                    2
```

```
Desync scenario:
  Jetson misses real 0xAA → starts reading mid-stream
  Sees 0xAA in payload at byte 73 → thinks it's a new frame start
  Reads next 2 bytes as length → gets nonsense (e.g. len=37543)
  Tries to read 37543 bytes → observes rubbish CRC
  CRC check fails → frame discarded (not passed to navigation!)
  Sequence number jump detected → Jetson logs desync event
  Next real 0xAA encountered → re-sync complete
```

Without CRC, the Jetson would have processed 37543 bytes of garbage sensor data — undetected.

**CRC16-CCITT is used (not CRC32) because:**
- 2 bytes of overhead (vs 4 for CRC32) — frame efficiency matters at 100 Hz
- Catches all 1, 2, 3-bit errors and most burst errors
- Fast to compute in C: single-pass table lookup, ~0.3 µs for 150 bytes
- Good enough for SPI distances (< 1m, low noise)

---

### 1.11 Frame desync and recovery — what happens when Jetson reboots mid-transfer

**The problem:** When the Jetson reboots or its spidev reader crashes, the STM32 just keeps filling buffers and swapping them. When the Jetson comes back, it starts reading SPI mid-cycle — it might be in the middle of byte 47 of a 150-byte frame.

**Three-layer recovery design:**

```
Layer 1: Sync byte (0xAA) — re-lock at frame boundary
  Jetson code: scan incoming bytes until 0xAA seen, then read header
  Handles: Jetson reconnect mid-stream, SPI framing drift

Layer 2: CRC16 — detect mid-frame garbage
  Jetson code: re-seek if CRC fails
  Handles: Partial frames, bit errors, length corruption

Layer 3: Sequence number — count dropped frames
  STM32: increments seq counter in header every frame
  Jetson: checks seq, logs drops (NOT an error if it's just 1-2 missed; IS an error if continuous)
  Handles: Performance monitoring, helps distinguish "Jetson missing frames" from "STM32 not sending"
```

**The sequence number is critical for OKS debugging.** When the Jetson's navigation stack misbehaves, you want to know: "Is the IMU data arriving intact?" A seq counter lets you answer "we dropped 0 frames" or "we dropped 3 frames per second" — immediately distinguishing SPI transport problems from nav algorithm problems.

---

### 1.12 DMA completion callback execution context — it runs as an ISR

**This is the source of subtle, hard-to-reproduce bugs.**

In Zephyr, the `spi_transceive_async()` callback fires from interrupt context. This means it is an ISR (Interrupt Service Routine), NOT a regular thread.

**ISR rules you must never break:**

| Action | In thread | In ISR |
|--------|----------|--------|
| `k_sem_give()` | ✓ Legal | ✓ Legal (this is how ISR notifies thread) |
| `k_sem_take()` | ✓ Legal | ✗ ILLEGAL — never blocks in ISR |
| `k_msleep()` | ✓ Legal | ✗ ILLEGAL — scheduling requires thread context |
| `LOG_INF()` | ✓ Legal | ✗ ILLEGAL — LOG uses a work queue (allocates, sleeps) |
| `printk()` | ✓ Legal | ✓ Legal (direct UART write, but slow) |
| Call `malloc()` | ✓ Legal | ✗ ILLEGAL — heap operations can block |
| Read/write global atomics | ✓ Legal | ✓ Legal |
| Update sequence counter | ✓ Legal | ✓ Legal |
| `k_work_submit()` | ✓ Legal | ✓ Legal (posts to work queue for later) |

**The pattern to follow:**

```c
// DMA complete callback — ISR CONTEXT
void spi_async_callback(const struct device *dev, int result, void *userdata)
{
    // DO: advance the active buffer index (atomic write)
    tx_active = 1 - tx_active;

    // DO: increment sequence counter (for desync detection)
    frame_seq++;

    // DO: signal the packer thread that buffer is now free
    k_sem_give(&frame_done_sem);  // ← this wakes up the thread, legally

    // DO NOT: call LOG_INF() ← ILLEGAL, will crash or corrupt log subsystem
    // DO NOT: call spi_transceive() ← ILLEGAL, will deadlock
    // DO NOT: call k_sleep() ← ILLEGAL, scheduling in ISR = undefined behavior
}
```

**Why does LOG_INF() crash in an ISR?** Zephyr's log subsystem routes messages through a work queue, which involves allocating a log message buffer and potentially blocking. In ISR context, blocking is illegal because the scheduler cannot preempt an ISR. The system will either hard fault or silently discard the log message and corrupt internal state.

---

## PART 2 — Annotated Code Reference

---

### 2.1 Buffer definitions — alignment and placement

```c
// comms/spi_slave.c

#include <zephyr/kernel.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/cache.h>          // SCB_CleanDCache_by_Addr etc.
#include "crc16.h"
#include "proto/sensor_frame.pb.h"

// ────────────────────────────────────────────────────────────────────────────
// BUFFER SIZING
// Maximum frame: nanopb-encoded SensorFrame (varies) + 5 byte framing overhead
//   (0xAA + 2-byte len + 2-byte CRC)
#define FRAME_MAX_PAYLOAD  256
#define FRAME_OVERHEAD     5    // header 3 + CRC 2
#define FRAME_MAX_BYTES    (FRAME_MAX_PAYLOAD + FRAME_OVERHEAD)

// ────────────────────────────────────────────────────────────────────────────
// DMA BUFFERS — must satisfy three constraints:
//
// 1. __aligned(32): Cortex-M7 cache line = 32 bytes.
//    SCB_CleanDCache_by_Addr() rounds DOWN to cache line boundary and rounds UP
//    to the next cache line. If your buffer starts at offset +4 within a cache
//    line, the flush may miss the first 4 bytes AND flush 28 bytes of the
//    *previous* struct (corrupting it). Always align to 32.
//
// 2. Placed in DMA-accessible SRAM: On STM32H7, DMA1/2 can only access D2
//    SRAM (AHB SRAM, addresses 0x30000000–0x3007FFFF).
//    D1 SRAM (DTCM, 0x20000000) is fast but NOT accessible by DMA1/2 — only
//    by MDMA or BDMA. Placing buffers in DTCM + using DMA1 = silent failure.
//    The .noinit_d2 section below tells the linker to place these in D2 SRAM.
//
// 3. Double-buffered: one buffer always under DMA ownership, one always
//    under CPU/packer ownership. They never overlap.
//
static uint8_t __aligned(32) __attribute__((section(".noinit_d2")))
    tx_buf[2][FRAME_MAX_BYTES];

// tx_active: index (0 or 1) of the buffer DMA is currently SENDING from.
// The OTHER index (1 - tx_active) is the idle buffer the packer writes into.
// Must be volatile: both the ISR and the packer thread read/write this.
// Atomic on ARM32: single 32-bit aligned word write is one STR instruction.
static volatile int tx_active = 0;

// tx_len: byte count of the frame currently in the active buffer.
// Updated before tx_active swap so the CS ISR always reads a consistent length.
static volatile uint16_t tx_len = 0;

// Semaphore signaling the packer thread that a DMA transfer completed.
// ISR calls k_sem_give() (legal from ISR). Packer thread calls k_sem_take().
static K_SEM_DEFINE(frame_done_sem, 0, 1);

// Frame sequence counter — incremented on every successful TX.
// Jetson monitors this to detect dropped frames.
static volatile uint32_t frame_seq = 0;
```

---

### 2.2 The CS GPIO interrupt handler — what to do and what NOT to do

```c
// GPIO spec for CS pin — obtained from devicetree at compile time
static const struct gpio_dt_spec cs_gpio =
    GPIO_DT_SPEC_GET(DT_NODELABEL(spi1_cs_mon), gpios);
    // spi1_cs_mon: a gpio-keys node exposing the CS line as a monitorable GPIO.
    // Separate from the SPI controller's NSS pin — we wire it to both the SPI
    // hardware AND a GPIO input so we can get an edge interrupt.

static struct gpio_callback cs_cb_data;  // Zephyr callback state struct

// ────────────────────────────────────────────────────────────────────────────
// CS FALLING EDGE ISR
// This fires when Jetson asserts CS (pulls it LOW).
//
// TIMING CONSTRAINT: the first SPI clock edge arrives 100ns–1µs after CS.
// You have approximately 0 useful time here.
//
// CORRECT behavior: do nothing meaningful. DMA was already armed in the
//   previous TX-complete callback. Just validate and return.
//
// WRONG behavior: arm DMA here. By the time your ISR body runs, the SPI
//   hardware has already sent the first 1–4 bytes as 0xFF.
//
static void cs_falling_isr(const struct device *dev,
                            struct gpio_callback *cb,
                            uint32_t pins)
{
    // OPTIONAL: sanity check — if DMA is somehow not armed, we can log a
    // counter to track pre-arm failures. But do NOT try to arm DMA here.
    // Arming takes >1µs and the SPI clock is already running.

    // This ISR's real job: detect CS falling so we know a transfer started.
    // Use this signal for watchdog reset or "host is alive" detection.
    // Otherwise: just return immediately.
    (void)dev;
    (void)cb;
    (void)pins;
}

// ────────────────────────────────────────────────────────────────────────────
// CS RISING EDGE ISR (optional but useful)
// Fires when Jetson deasserts CS — transfer is over.
// Ideal place to count completed transfers for throughput monitoring.
static void cs_rising_isr(const struct device *dev,
                           struct gpio_callback *cb,
                           uint32_t pins)
{
    (void)dev; (void)cb; (void)pins;
    // Just a counter increment — purely diagnostic, not in critical path.
}

void spi_slave_gpio_init(void)
{
    gpio_pin_configure_dt(&cs_gpio, GPIO_INPUT);

    gpio_init_callback(&cs_cb_data, cs_falling_isr, BIT(cs_gpio.pin));
    gpio_add_callback(cs_gpio.port, &cs_cb_data);

    // Trigger on BOTH edges: falling = transfer start, rising = transfer end
    gpio_pin_interrupt_configure_dt(&cs_gpio, GPIO_INT_EDGE_BOTH);
}
```

---

### 2.3 `spi_transceive_async()` — every parameter explained

```c
static const struct device *spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi1));

// SPI slave configuration — note key differences from master config
static const struct spi_config slave_cfg = {
    .frequency = 0,
    // ↑ SLAVE SETS FREQUENCY TO 0: the slave does NOT generate the clock.
    //   Setting a non-zero frequency is ignored in slave mode, but 0 makes
    //   the intent explicit. (Putting 10000000 here does nothing harmful but
    //   is misleading to readers.)

    .operation =
        SPI_WORD_SET(8)         |   // 8-bit words (standard)
        SPI_OP_MODE_SLAVE       |   // ← slave mode: wait for master clock
        SPI_TRANSFER_MSB        |   // MSB first (check with Jetson spidev config!)
        SPI_MODE_CPOL           |   // CPOL=1: clock idle HIGH
        SPI_MODE_CPHA,              // CPHA=1: sample on second edge
        // ↑ Mode 3 (CPOL=1, CPHA=1): common choice matching Jetson default.
        //   Check your Jetson spidev mode with: cat /sys/bus/spi/.../of_node/mode
        //   Both sides must agree or every byte is bit-rotated garbage.

    .cs = {                         // Slave does not control CS gpio —
        .gpio = { .port = NULL },   // leave NULL here. Master controls CS.
        .delay = 0,
    },
    .slave = 0,                     // slave index (for multi-slave, usually 0)
};

// Buffer descriptors — Zephyr SPI API uses scatter/gather list
static struct spi_buf tx_spi_buf;   // points to tx_buf[tx_active]
static struct spi_buf rx_spi_buf;   // receive buffer (we usually don't care about RX)
static struct spi_buf_set tx_set = { .buffers = &tx_spi_buf, .count = 1 };
static struct spi_buf_set rx_set = { .buffers = &rx_spi_buf, .count = 1 };

// RX buffer — we need one even if we don't care about incoming bytes.
// The SPI hardware is always full-duplex. We just ignore the received data.
static uint8_t __aligned(32) rx_discard_buf[FRAME_MAX_BYTES];

// ────────────────────────────────────────────────────────────────────────────
// ARM DMA FOR NEXT TRANSFER
// Call this from the TX-complete callback (NOT from CS ISR).
static void arm_spi_dma(void)
{
    // Point TX buffer at the currently active frame
    tx_spi_buf.buf = tx_buf[tx_active];
    tx_spi_buf.len = tx_len;        // exact byte count

    // Point RX buffer at the discard area
    rx_spi_buf.buf = rx_discard_buf;
    rx_spi_buf.len = tx_len;        // RX and TX must be same length for SPI

    // Flush D-cache BEFORE arming DMA
    // ↓ This is the critical line. Without it: DMA reads stale SRAM → Jetson
    //   receives garbage while GDB shows perfect data (the lying librarian).
    sys_cache_data_flush_range(tx_buf[tx_active], tx_len);
    // ↑ Zephyr 3.x abstraction for SCB_CleanDCache_by_Addr.
    //   On M0/M3/M4 this is a no-op (no D-cache) — safe on all targets.

    // Launch async SPI transfer — this call is non-blocking
    // Returns immediately; callback fires when Jetson completes the transfer.
    int ret = spi_transceive_async(
        spi_dev,                // device: obtained from devicetree
        &slave_cfg,             // config: slave mode, mode 3, etc.
        &tx_set,                // TX buffer set: what we send
        &rx_set,                // RX buffer set: where incoming bytes go (discard)
        &spi_async_cb,          // callback: fired from ISR when transfer completes
        NULL                    // userdata: passed to callback (NULL here)
    );

    if (ret != 0) {
        // arm failed — this is serious. Increment error counter.
        // Do NOT call LOG_ERR here if called from ISR context.
        // If called from thread context: LOG_ERR("SPI arm failed: %d", ret);
        arm_error_count++;
    }
}
```

---

### 2.4 The DMA complete callback — legal operations only

```c
// ────────────────────────────────────────────────────────────────────────────
// SPI ASYNC COMPLETION CALLBACK
//
// EXECUTION CONTEXT: ISR — called from hardware interrupt, not a thread.
// Rules: no blocking, no LOG_INF, no malloc, no k_sleep, no k_sem_take.
// Allowed: k_sem_give, atomic writes, increment counters, k_work_submit.
//
static void spi_async_cb(const struct device *dev,
                         int result,
                         void *userdata)
{
    (void)dev;
    (void)userdata;

    if (result != 0) {
        // SPI transfer error (cable unplugged, invalid state, etc.)
        tx_error_count++;           // ← atomic (single 32-bit write on ARM32)
        // Re-arm anyway — keep trying. The next transfer may succeed.
    } else {
        frame_seq++;                // ← count successful frames sent
    }

    // Immediately re-arm DMA for the NEXT transfer.
    // This is the critical moment: DMA is now armed and waiting.
    // CS can fire any time after this returns — and we'll be ready.
    arm_spi_dma();                  // ← points DMA at tx_buf[tx_active]

    // Signal the packer thread that a transfer completed.
    // The packer thread blocks on this semaphore, then:
    //   1. Reads tx_active to know which buffer DMA is CURRENTLY using
    //   2. Writes new frame into the OTHER buffer
    //   3. Swaps tx_active pointer
    k_sem_give(&frame_done_sem);    // ← legal from ISR: unblocks thread

    // DO NOT:
    //   LOG_INF("frame sent");     ← ILLEGAL: work queue, alloc, potential block
    //   spi_transceive(spi_dev,…); ← ILLEGAL: blocking API in ISR
    //   k_sleep(K_MSEC(1));        ← ILLEGAL: no scheduling in ISR
}
```

---

### 2.5 The packer thread main loop — encode → flush cache → swap → signal

```c
// ────────────────────────────────────────────────────────────────────────────
// PACKER THREAD
// Runs at priority 3 (below main thread, above idle).
// Wakes when DMA signals frame_done_sem. Encodes next frame into idle buffer.
// This is the only thread that writes to tx_buf — no locking needed.
//
void spi_packer_thread(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1); ARG_UNUSED(arg2); ARG_UNUSED(arg3);

    // Arm DMA once at startup — before first CS edge arrives
    tx_active = 0;
    tx_len    = 0;
    memset(tx_buf[0], 0x00, sizeof(tx_buf[0]));  // zero initial frame
    arm_spi_dma();                                // DMA armed; waiting for CS

    while (1) {
        // ── Step 1: wait for previous DMA transfer to complete ──────────────
        // This blocks until the ISR calls k_sem_give(&frame_done_sem).
        // Thread switches to sleeping — CPU free for other threads.
        k_sem_take(&frame_done_sem, K_FOREVER);
        // ↑ K_FOREVER: if DMA never completes (STM32 bug, Jetson not polling),
        //   thread blocks forever. If this is a concern, use K_MSEC(50) and
        //   log a watchdog warning.

        // ── Step 2: determine idle buffer index ─────────────────────────────
        // tx_active is the buffer DMA is currently pointing at (just re-armed).
        // We write into the OTHER buffer — the one DMA is NOT reading.
        int write_idx = 1 - tx_active;
        // ↑ On ARM32, reading tx_active is atomic (one LDR instruction).
        //   write_idx is stable: arm_spi_dma() inside the ISR cannot change
        //   tx_active between this read and when we finish writing write_idx.

        // ── Step 3: encode nanopb payload into idle buffer (after framing) ──
        uint8_t *payload_start = &tx_buf[write_idx][3]; // skip 3-byte header
        pb_ostream_t stream = pb_ostream_from_buffer(
            payload_start, FRAME_MAX_PAYLOAD);

        SensorFrame frame = get_latest_sensor_frame();  // reads shared data
        bool encode_ok = pb_encode(&stream, SensorFrame_fields, &frame);

        uint16_t payload_len = (uint16_t)stream.bytes_written;

        if (!encode_ok || payload_len == 0) {
            // Encoding failed — send a zero-length frame with error flag set
            payload_len = 0;
        }

        // ── Step 4: build frame header ──────────────────────────────────────
        tx_buf[write_idx][0] = 0xAA;                    // sync byte
        tx_buf[write_idx][1] = (payload_len >> 8) & 0xFF;
        tx_buf[write_idx][2] =  payload_len       & 0xFF;

        // ── Step 5: append CRC16 ─────────────────────────────────────────────
        uint16_t crc = crc16_ccitt(payload_start, payload_len);
        tx_buf[write_idx][3 + payload_len]     = (crc >> 8) & 0xFF;
        tx_buf[write_idx][3 + payload_len + 1] =  crc       & 0xFF;

        uint16_t total_len = 3 + payload_len + 2;  // header + payload + CRC

        // ── Step 6: flush D-cache for entire frame ───────────────────────────
        // This is the cache coherency fix. Write CPU cache → SRAM so DMA
        // sees fresh data when it reads this buffer next transfer.
        // Must happen BEFORE tx_len and tx_active are updated.
        sys_cache_data_flush_range(tx_buf[write_idx], total_len);
        // Align consideration: tx_buf[write_idx] is __aligned(32) so the
        // flush covers exactly the right range with no off-by-32 errors.

        // ── Step 7: atomic buffer swap ───────────────────────────────────────
        // Update tx_len first, then tx_active.
        // Order matters: arm_spi_dma() in the ISR reads tx_active to find the
        // buffer, then reads tx_len for the byte count. If we set tx_active
        // first, the ISR might catch a stale tx_len value.
        tx_len   = total_len;   // set length before making buffer active
        tx_active = write_idx;  // ← ATOMIC: single 32-bit write (one STR)
        // After this write, the next arm_spi_dma() call in the ISR will use
        // the new buffer and new length. The previous buffer is now "idle"
        // and safe for us to overwrite in the next iteration.
    }
}

// Static thread definition
K_THREAD_DEFINE(spi_packer, 2048, spi_packer_thread, NULL, NULL, NULL, 3, 0, 0);
//                           ↑     ↑                             ↑
//                   stack bytes  entry fn                  priority 3
// Stack 2048: nanopb encoding can use 400–600 bytes for message struct + stream.
// CRC + framing + memcpy uses another ~100 bytes.
// 2048 gives ~1000 bytes headroom — check with k_thread_stack_space_get() in dev.
```

---

### 2.6 The double-buffer swap — why atomicity matters

```c
// ── CORRECT: update length before pointer swap ──────────────────────────────
tx_len   = total_len;    // Step A: length for new buffer
tx_active = write_idx;   // Step B: NOW make it active (ISR sees consistent state)

// WHY the order matters:
// ISR calls arm_spi_dma() which does:
//   tx_spi_buf.buf = tx_buf[tx_active];      ← reads tx_active (A = write_idx now)
//   tx_spi_buf.len = tx_len;                 ← reads tx_len
//
// If you set tx_active BEFORE tx_len:
//   ISR reads tx_active = write_idx   ← correct new buffer
//   ISR reads tx_len    = OLD_LEN     ← stale length! Sends too many/few bytes!
//
// If you set tx_len BEFORE tx_active:
//   ISR reads tx_active = OLD_IDX     ← points to old buffer (who cares, it's armed)
//   ISR reads tx_len    = NEW_LEN     ← new length for old buffer (wrong but harmless)
//   Next ISR call will get both new   ← correct by next transfer (one frame error max)
//
// The safest invariant: always write the length before swapping the pointer.


// ── WRONG: naive Python-like swap ───────────────────────────────────────────
// This would be wrong because there's no "atomic pair" swap on ARM32:
//
// tx_active = write_idx;   ← ISR could read tx_active here (sees new idx)
// tx_len = total_len;      ← ISR also reads tx_len here (sees old len) = CORRUPT FRAME
//
// Always update length first, then pointer.
```

---

### 2.7 Wrong vs correct: arm DMA in CS ISR vs arm in TX-complete callback

```c
// ══════════════════════════════════════════════════════════════════════════
// WRONG APPROACH — Arm DMA inside the CS falling-edge interrupt
// ══════════════════════════════════════════════════════════════════════════

static void cs_falling_isr_WRONG(const struct device *dev,
                                  struct gpio_callback *cb,
                                  uint32_t pins)
{
    // ← You are here at t=0ns (CS just went low)
    // ← First SPI clock edge arrives at t=100ns to t=1000ns
    //
    // This function body takes:
    //   - IRQ entry overhead:               ~25 ns
    //   - Possible IRQ masking delay:       0–10,000 ns (if critical section active)
    //   - arm_spi_dma() execution:          ~500–2000 ns
    //
    // Total: 525–12,025 ns
    //
    // At 10 MHz: 1 bit = 100ns, 1 byte = 800ns
    // If arm takes 1600ns: first 2 bytes already sent as 0xFF before DMA is armed
    //
    arm_spi_dma();  // ← RACE: SPI clock may already be running!

    // SYMPTOM: First 1–4 bytes of every frame are 0xFF (or 0x00)
    //          Rest of frame is correct
    //          Error is CONSISTENT (not intermittent)
    //          No error flags from SPI hardware
    //          Adding LOG_INF here makes bug WORSE (ISR takes longer → more 0xFF bytes)
}

// ══════════════════════════════════════════════════════════════════════════
// CORRECT APPROACH — Arm DMA when PREVIOUS transfer completes
// ══════════════════════════════════════════════════════════════════════════

static void spi_async_cb_CORRECT(const struct device *dev,
                                  int result,
                                  void *userdata)
{
    // ← You are here when the PREVIOUS transfer just finished (CS went HIGH)
    // ← Next CS edge could arrive in 1ms, 5ms, 10ms — we don't know
    //
    // This is the RIGHT time to arm. We have microseconds to milliseconds of
    // margin. The DMA will be armed and waiting well before CS fires again.
    //
    frame_seq++;
    arm_spi_dma();      // ← re-arm immediately after previous TX completes
    k_sem_give(&frame_done_sem);

    // TIMING: arm_spi_dma() is called here at t=T_prev_complete
    //         CS fires at t=T_prev_complete + (some interval, often 10ms)
    //         DMA is armed for 10ms before it needs to respond
    //         First byte sent correctly: guaranteed
}
```

---

### 2.8 CRC16 framing — complete frame assembly

```c
// ────────────────────────────────────────────────────────────────────────────
// FRAME FORMAT:
//   Byte 0:     0xAA (sync)
//   Bytes 1–2:  payload length, big-endian uint16
//   Bytes 3..N: nanopb payload
//   Bytes N+1, N+2: CRC16-CCITT over bytes 3..N (payload only, not header)
//
// CRC is computed over payload only (not the 3-byte header).
// This simplifies resync: Jetson finds 0xAA, reads 2 bytes for length,
// reads `length` bytes, verifies CRC. No need to include header in CRC
// because the length field is self-consistent (a corrupted length will
// produce a wrong byte count which CRC will catch anyway).
//
void build_spi_frame(uint8_t *frame_buf,
                     const uint8_t *payload,
                     uint16_t payload_len,
                     uint16_t *total_len_out)
{
    // ── Header ────────────────────────────────────────────────────────────
    frame_buf[0] = 0xAA;
    frame_buf[1] = (payload_len >> 8) & 0xFF;  // length high byte
    frame_buf[2] =  payload_len       & 0xFF;  // length low byte

    // ── Payload ───────────────────────────────────────────────────────────
    memcpy(&frame_buf[3], payload, payload_len);

    // ── CRC16-CCITT ───────────────────────────────────────────────────────
    // Initial value 0xFFFF, polynomial 0x1021, no final XOR — standard CCITT
    uint16_t crc = crc16_ccitt(&frame_buf[3], payload_len);
    // ↑ crc16_ccitt() from Zephyr's <zephyr/sys/crc.h>:
    //   uint16_t crc16_ccitt(uint16_t seed, const uint8_t *src, size_t len);
    //   Call as: crc16_ccitt(0xFFFF, payload_ptr, len)
    //   Or implement with a 256-entry lookup table for ~0.3µs per frame.

    frame_buf[3 + payload_len]     = (crc >> 8) & 0xFF;  // CRC high byte
    frame_buf[3 + payload_len + 1] =  crc       & 0xFF;  // CRC low byte

    *total_len_out = 3 + payload_len + 2;  // total frame size

    // Bounds check — defensive, should never trigger if caller is correct
    __ASSERT(*total_len_out <= FRAME_MAX_BYTES,
             "Frame too large: %u bytes (max %u)",
             *total_len_out, FRAME_MAX_BYTES);
}
```

---

### 2.9 Frame counter and error counter layout in the header

```c
// ────────────────────────────────────────────────────────────────────────────
// Extended frame header layout (recommended for production):
//
//   Byte 0:     0xAA (sync byte)
//   Bytes 1–2:  payload length, big-endian uint16
//   Bytes 3–6:  sequence number, big-endian uint32 (Jetson detects dropped frames)
//   Bytes 7–8:  error flags (DMA errors, encode failures, watchdog timeouts)
//   Bytes 9..N: nanopb payload
//   Bytes N+1, N+2: CRC16 over all bytes from 3 to N (seq + flags + payload)
//
// Error flag bits (bytes 7–8):
#define ERR_DMA_ARM_FAIL     BIT(0)  // arm_spi_dma() returned non-zero
#define ERR_ENCODE_FAIL      BIT(1)  // pb_encode() returned false
#define ERR_WATCHDOG         BIT(2)  // packer took >15ms (missed a cycle)
#define ERR_CACHE_SKIP       BIT(3)  // cache flush was skipped (shouldn't happen)

// Writing the extended header:
static void build_extended_header(uint8_t *buf, uint16_t payload_len)
{
    uint32_t seq = frame_seq;  // snapshot before encoding (ISR might increment)
    uint16_t errs = pending_error_flags;
    pending_error_flags = 0;   // clear for next frame

    buf[0] = 0xAA;
    buf[1] = (payload_len >> 8) & 0xFF;
    buf[2] =  payload_len       & 0xFF;
    buf[3] = (seq >> 24) & 0xFF;
    buf[4] = (seq >> 16) & 0xFF;
    buf[5] = (seq >>  8) & 0xFF;
    buf[6] =  seq        & 0xFF;
    buf[7] = (errs >> 8) & 0xFF;
    buf[8] =  errs       & 0xFF;
}
```

---

## PART 3 — Gotcha Table

---

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| **First 1–4 bytes of every frame are 0xFF or 0x00, rest is correct** | DMA armed inside CS falling-edge ISR — too late, SPI clock already running when DMA is configured | Logic analyzer: measure time from CS falling edge to first incorrect byte. If always ≤ 4 bytes wrong, this is timing. Scope DMA_TC interrupt pin: fires AFTER CS? Wrong. | Arm DMA in the TX-complete callback (`spi_async_cb`), not in `cs_falling_isr`. DMA must be pre-armed before CS fires. |
| **GDB shows correct values in tx_buf[], but Jetson receives garbage. "Correct" in GDB, wrong over wire.** | D-cache coherency: CPU writes went to D-cache only; SRAM (which DMA reads) still has stale data. GDB forces a cache line fill when it reads, hiding the bug. | Pattern: first 32 bytes correct (one cache line was evicted), bytes 32+ garbage. Toggle a GPIO before and after `SCB_CleanDCache_by_Addr` to confirm the flush call exists. Put a `__asm__ volatile("dsb sy")` before DMA start; if that fixes it, it was a cache issue. | Add `sys_cache_data_flush_range(tx_buf[idx], len)` BEFORE `spi_transceive_async()`. Ensure buffers are `__aligned(32)`. Consider non-cacheable SRAM section for permanent fix. |
| **Bytes are correct but in the WRONG buffer (neighbors corrupted)**. Random bytes appearing in adjacent structs. | Buffer not `__aligned(32)`. `SCB_CleanDCache_by_Addr` rounds addresses to 32-byte boundaries. An unaligned buffer causes the flush to cover the wrong range — including adjacent memory. | Add `static_assert(sizeof(tx_buf[0]) % 32 == 0, "not cache-line aligned")`. Print `(uint32_t)tx_buf[0] % 32` at startup — should be 0. | Declare buffers with `__aligned(32)`. Also round up buffer size to multiple of 32: `(FRAME_MAX_BYTES + 31) & ~31`. |
| **All bytes wrong — MISO shows bit-pattern that looks like a shift of the correct data** | Wrong SPI mode (CPOL/CPHA mismatch). When modes differ, bits are sampled at the wrong clock edge, rotating or inverting the bit stream. | Logic analyzer: capture raw MISO. Compare bit pattern to expected. The bits are almost all there but shifted by half a clock cycle. Try all 4 SPI modes systematically. | Match CPOL/CPHA to what Jetson expects. On Jetson Orin, default is Mode 0. Set `SPI_MODE_CPOL | SPI_MODE_CPHA` in slave config only if Jetson explicitly uses Mode 3. |
| **DMA works but is extremely slow (CPU usage 80%+)** | `CONFIG_SPI_STM32_DMA=y` missing from prj.conf. SPI driver fell back to CPU polling mode silently. | Add `printk("SPI DMA enabled: %d\n", IS_ENABLED(CONFIG_SPI_STM32_DMA))` at startup. Use `perf stat` equivalent: toggle GPIO in packer loop, measure pulse width with logic analyzer. Should be <500µs with DMA. | Add `CONFIG_SPI_STM32_DMA=y` to prj.conf. Also needs `CONFIG_DMA=y`. Rebuild. |
| **SPI transfers work in testing, fail intermittently with nanopb message that has many zero fields** | nanopb / proto3 omits default-value (zero) fields during encode. Message encodes to fewer bytes when GPS drops fix or all floats are 0. If your Jetson expects fixed-length frames, it over-reads the frame boundary. | Add `LOG_DBG("encoded %u bytes", stream.bytes_written)` in packer. Send artificial all-zero sensor data and check if Jetson frame length errors spike. | Use explicit `payload_len` field in proto header. Jetson reads exactly `payload_len` bytes from after the sync header, not a fixed size. |
| **DMA works once, then all subsequent transfers send 0xFF** | Buffer placed in DTCM (D1 CPU-local SRAM, address 0x20000000). DMA1/2 cannot access DTCM on STM32H7. First transfer may succeed by accident or from cache; thereafter DMA reads a region it has no physical path to. | Check linker map: if tx_buf lands near 0x20000000, it's in DTCM. DMA for SPI on STM32H7 needs D2 SRAM (~0x30000000). | Add `__attribute__((section(".noinit_d2")))` to buffer declarations. Add custom linker section for D2 SRAM in your linker script or board definition: `.noinit_d2 (NOLOAD) : { *(.noinit_d2) } > SRAM2` |
| **SPI transfer never completes (spi_transceive_async() callback never fires)** | Jetson never asserts CS, OR SPI slave configured as master by mistake, OR `CONFIG_SPI_SLAVE=y` missing. For slave mode to work, Zephyr needs the slave API enabled. | Check prj.conf for `CONFIG_SPI_SLAVE=y`. Check slave_cfg: `.operation` must include `SPI_OP_MODE_SLAVE`. Verify CS GPIO wired correctly (use logic analyzer to confirm Jetson is asserting CS). | Add `CONFIG_SPI_SLAVE=y` to prj.conf. Verify `SPI_OP_MODE_SLAVE` in config. Check CS polarity (should be active-low on STM32: `GPIO_ACTIVE_LOW`). |
| **nanopb encoded into a local stack variable, then that address passed to DMA** | Stack is in DTCM (fast CPU-local SRAM, not DMA-accessible on H7). DMA will read from an inaccessible address, causing a bus error or silent garbage. | Run under GDB: `info frame` during encoding shows local buffer at an address starting with 0x20xxxxxx = DTCM. | Always encode into `tx_buf[]` (the DMA-accessible buffer in D2 SRAM), never into a local stack variable first. |
| **CRC passes but Jetson nav stack drifts over time (subtle errors)** | LSB/MSB byte order mismatch between STM32 CRC16 and Jetson CRC16 implementations. Some libraries swap the byte order; CRC happens to match on uniform bytes but fails on others. | Log raw frame bytes in hex on both sides for the same transfer. Compute CRC manually. If Jetson CRC starts at seed=0x0000 and STM32 at seed=0xFFFF, they will always disagree. | Standardize on CRC16-CCITT with seed=0xFFFF (standard CCITT). Verify both implementations: `crc16([0x01, 0x02, 0x03, 0x04])` should give the same hex value on both sides. |
| **`spi_transceive_async()` returns -ENOTSUP or -EINVAL** | Async SPI API not supported for this SPI instance, OR missing DMA channels in devicetree, OR `dmas =` and `dma-names =` not configured for the SPI node. | `west build` with `CONFIG_SPI_LOG_LEVEL_DBG=y`. Check build output for DMA init messages. Check if `dmas = <&dma1 3 ...>` is in your .overlay. | Add `dmas = <&dma1 3 3 0x0>, <&dma1 2 3 0x0>; dma-names = "tx", "rx";` to your SPI node in the overlay. Values depend on your STM32H7 DMA request mapping table (RM0433). |

---

## Quick Reference Card

---

```
SPI SLAVE + DMA + DOUBLE BUFFER — Critical Rules

1. PRE-ARM DMA, NEVER RE-ARM IN CS ISR
   • Arm DMA in TX-complete callback (arm on previous done)
   • CS ISR must be a no-op (too late to arm)
   • Test: logic analyzer — DMA TC interrupt should fire BEFORE CS goes high

2. D-CACHE FLUSH BEFORE EVERY TX
   • sys_cache_data_flush_range(buf, len) BEFORE spi_transceive_async()
   • "GDB shows correct, Jetson gets garbage" = ALWAYS this bug
   • Symptom fingerprint: first 32 bytes OK, rest garbage (= one cache line correct)

3. BUFFER MUST BE __aligned(32)
   • Cache line = 32 bytes on Cortex-M7
   • Misalignment → flush covers wrong range → adjacent memory corrupted
   • Static assert: _Static_assert((uintptr_t)tx_buf % 32 == 0, "align!")

4. BUFFER MUST BE IN DMA-ACCESSIBLE SRAM (STM32H7)
   • DTCM (0x20000000): CPU-only, DMA1/2 CANNOT access → silent failure
   • D2 SRAM (0x30000000): DMA-accessible → use section(".noinit_d2")

5. DMA CALLBACK IS AN ISR — NO BLOCKING
   • k_sem_give()   ✓    LOG_INF()       ✗
   • counter++      ✓    k_sem_take()    ✗
   • arm_spi_dma()  ✓    spi_transceive() ✗

6. DOUBLE BUFFER SWAP: LENGTH BEFORE POINTER
   • tx_len = new_len;       ← first
   • tx_active = write_idx;  ← second (atomic)
   • Reversing this risks ISR catching new pointer with old length

7. SPI MODE MUST MATCH JETSON
   • Check with: cat /sys/kernel/debug/spi0/... or spidev test
   • Default Jetson: Mode 0 (CPOL=0, CPHA=0)
   • Wrong mode: all bits wrong, no errors reported

8. KCONFIG CHECKLIST
   CONFIG_SPI=y
   CONFIG_SPI_SLAVE=y
   CONFIG_SPI_STM32_DMA=y   ← SPI WITHOUT THIS = CPU polling (slow)
   CONFIG_DMA=y
   CONFIG_CACHE=y            ← for sys_cache_data_flush_range on M7
```

```
TIMING BUDGET @ 100Hz (10ms frame interval)
  SPI transfer (150 bytes @ 10MHz):   120 µs  (12% of 1ms)
  DMA setup overhead:                   1 µs
  Cache flush (150 bytes):              0.5 µs
  nanopb encode:                       10–50 µs
  packer thread overhead:              ~100 µs total
  ────────────────────────────────────────────
  Budget used:                        ~200 µs / 10,000 µs = 2%
  DMA active time DMA in ISR:          ~1.5 µs (arm + callback)
  Leaves 9.8ms for IMU read, fusion, CAN, shell, etc.
```

```
FRAME LAYOUT (wire format, byte-exact)
  [0xAA][LEN_HI][LEN_LO][SEQ_3][SEQ_2][SEQ_1][SEQ_0][ERR_HI][ERR_LO][...payload...][CRC_HI][CRC_LO]
     1      1       1       1      1      1       1      1       1      0..256            1       1
  Header = 9 bytes   │   Payload = 0..256 bytes   │   Footer = 2 bytes
  CRC covers bytes 3..N (seq + flags + payload). Does NOT cover first 3 bytes.
  Total frame: 11 + payload_len bytes. Max = 267 bytes.
```

```
DIAGNOSTIC COMMANDS (logic analyzer + GDB)

Logic analyzer: trigger = CS falling, decode = SPI
  • Channel 0: CS
  • Channel 1: CLK (set speed > 2× SPI frequency)
  • Channel 2: MISO (STM32 TX to Jetson)
  • Channel 3: DMA TC GPIO toggle (optional: tog pin at callback start/end)

GDB (JLink/OpenOCD):
  (gdb) p tx_buf[0]          ← reads via CPU (fills cache, shows cache value)
  (gdb) x/64xb tx_buf[0]    ← same issue: GDB uses CPU reads
  (gdb) monitor halt
  (gdb) set $addr = &tx_buf[0]
        Check SRAM directly via memory-mapped access with DMA active for truth

Jetson (spidev_test):
  spidev_test -D /dev/spidev0.0 -s 10000000 -v -p '\xAA\x00\x00'
```

