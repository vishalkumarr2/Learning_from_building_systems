# DMA — Direct Memory Access

## The Core Problem

Without DMA, the CPU copies every byte manually:

```c
// Sending 256 bytes over SPI — CPU-driven (polling)
for (int i = 0; i < 256; i++) {
    SPI->DR = buffer[i];           // CPU writes byte to SPI hardware register
    while (!(SPI->SR & TXE));      // CPU spins waiting for SPI hardware to be ready
}
// CPU is 100% occupied for the entire transfer — can't do anything else
```

At 10MHz SPI, 256 bytes = ~205 microseconds of CPU doing nothing but waiting.
At 100Hz (10ms frame), that's **2% of your CPU budget** just on copying bytes.

---

## What DMA Does

DMA is a **separate hardware engine** on the chip. You point it at source + destination + count, say "go", and the CPU is free.

```
Without DMA:
  CPU ──► copies byte 0 to SPI_DR ──► waits ──► byte 1 ──► waits ──► ...
  (CPU blocked entire time)

With DMA:
  CPU: "DMA: copy buffer[0..255] → SPI_DR, tell me when done" (takes ~200ns)
  CPU: goes and runs other threads
  DMA: quietly feeds bytes to SPI at wire speed
  DMA: fires interrupt when done
  CPU: handles completion ISR (takes ~500ns)
```

---

## Physically What Happens

```
┌─────────────┐  system bus  ┌─────────┐  peripheral bus  ┌──────────────┐
│    SRAM     │ ────────────►│   DMA   │ ────────────────► │  SPI_DR reg  │──► MOSI pin
│  (buffer)   │              │ engine  │                   │  (hardware)  │
└─────────────┘              └─────────┘                   └──────────────┘
                                  │
                        No CPU involved at all
```

The DMA controller has registers you configure:
- **Source address (M0AR)** — where to read (your buffer in SRAM)
- **Destination address (PAR)** — where to write (SPI_DR — fixed)
- **Count (NDTR)** — how many bytes to transfer
- **Increment flags** — auto-increment source address; keep destination fixed

---

## DMA Transfer Modes

### Mode 1: Memory → Peripheral (TX) — sending data out

```c
// Example: send buffer over SPI
DMA1_Stream3->PAR  = (uint32_t)&SPI1->DR;  // dest: SPI data register (fixed)
DMA1_Stream3->M0AR = (uint32_t)tx_buf;      // src: your buffer (auto-increments)
DMA1_Stream3->NDTR = 256;                   // count: 256 bytes
DMA1_Stream3->CR  |= DMA_SxCR_DIR_0;       // direction: memory to peripheral
DMA1_Stream3->CR  |= DMA_SxCR_EN;          // GO
```

### Mode 2: Peripheral → Memory (RX) — receiving data in

```c
// Example: receive into buffer from SPI
DMA1_Stream2->PAR  = (uint32_t)&SPI1->DR;  // src: SPI data register
DMA1_Stream2->M0AR = (uint32_t)rx_buf;      // dest: your buffer
DMA1_Stream2->NDTR = 256;
// Direction default (0) = peripheral to memory
DMA1_Stream2->CR  |= DMA_SxCR_EN;
```

### Mode 3: Memory → Memory — fast memcpy

```c
// Hardware-accelerated copy between two SRAM regions
DMA2_Stream0->PAR  = (uint32_t)src_buf;
DMA2_Stream0->M0AR = (uint32_t)dst_buf;
DMA2_Stream0->NDTR = 1024;
DMA2_Stream0->CR  |= DMA_SxCR_MEM2MEM;  // M2M mode
DMA2_Stream0->CR  |= DMA_SxCR_EN;
```

Transfer width options (source and destination can differ):
```c
// 8-bit bytes (default)  — normal for SPI/UART
// 16-bit half-words       — I2S audio, ADC
// 32-bit words            — fast memcpy, ADC burst
DMA1_Stream0->CR |= DMA_SxCR_MSIZE_1;   // memory width: 16-bit
DMA1_Stream0->CR |= DMA_SxCR_PSIZE_0;   // peripheral width: 8-bit
// DMA automatically packs/unpacks
```

---

## DMA Interrupts

```c
// DMA interrupt flags (fire an ISR when set):
DMA_IT_TC   // Transfer Complete  — all N bytes done               ← use this most
DMA_IT_HT   // Half Transfer      — N/2 bytes done (start processing early)
DMA_IT_TE   // Transfer Error     — bus error, invalid address     ← always handle
DMA_IT_FE   // FIFO Error

// Enable in CR before starting DMA
DMA1_Stream3->CR |= DMA_SxCR_TCIE;   // enable TC interrupt
DMA1_Stream3->CR |= DMA_SxCR_TEIE;   // enable TE interrupt

// ISR
void DMA1_Stream3_IRQHandler(void)
{
    if (DMA1->LISR & DMA_LISR_TCIF3) {
        DMA1->LIFCR = DMA_LIFCR_CTCIF3;  // clear flag (must do this!)
        on_spi_tx_complete();
    }
    if (DMA1->LISR & DMA_LISR_TEIF3) {
        DMA1->LIFCR = DMA_LIFCR_CTEIF3;
        log_error("DMA bus error");
    }
}
```

---

## Circular Mode — Free-Running Buffer

Normal mode: transfer N bytes → stop → CPU must restart.
Circular mode: wrap back to start automatically — runs **forever**.

Perfect for ADC, microphone, UART receive — anything continuous.

```c
DMA1_Stream0->CR |= DMA_SxCR_CIRC;   // enable circular mode
DMA1_Stream0->CR |= DMA_SxCR_HTIE;   // half-transfer interrupt
DMA1_Stream0->CR |= DMA_SxCR_TCIE;   // full-transfer interrupt
DMA1_Stream0->NDTR = 256;             // total buffer size
DMA1_Stream0->M0AR = (uint32_t)audio_buf;
DMA1_Stream0->CR  |= DMA_SxCR_EN;    // starts forever
```

```
DMA fills:  [0─────────127][128──────255][0─────────127][128──────255]...
                ↑HT ISR               ↑TC ISR
CPU processes:       [128──────255]          [0─────────127]

CPU always processes the HALF that DMA is NOT currently writing.
Zero gaps, zero dropped samples.
```

**Real use**: microphone input (I2S), GPS UART RX, encoder pulse counting.

---

## Hardware Double Buffer Mode (DBM)

STM32 DMA can automatically alternate between two buffers — no software swap needed:

```c
DMA1_Stream1->CR  |= DMA_SxCR_DBM;          // enable double buffer mode
DMA1_Stream1->M0AR = (uint32_t)buf_A;        // first buffer
DMA1_Stream1->M1AR = (uint32_t)buf_B;        // second buffer
DMA1_Stream1->NDTR = 512;
DMA1_Stream1->CR  |= DMA_SxCR_EN;

// DMA fills buf_A completely, fires TC, switches to buf_B automatically
// CPU can safely read buf_A while DMA fills buf_B
// No software pointer swap needed

void DMA1_Stream1_IRQHandler(void)
{
    if (TC flag) {
        // CT bit tells you which buffer DMA just FINISHED (now writing the other)
        if (DMA1_Stream1->CR & DMA_SxCR_CT) {
            process_buffer(buf_A);  // DMA moved to buf_B, so buf_A is safe
        } else {
            process_buffer(buf_B);  // DMA moved to buf_A, so buf_B is safe
        }
    }
}
```

---

## The Cache Coherency Bug (Cortex-M7 / Cortex-A)

**Affects**: STM32H7, STM32F7, Jetson, Raspberry Pi CM4 — any chip with D-cache enabled.

```
The bug:
  1. CPU writes new data to tx_buf        → goes into L1 D-cache (fast path)
                                            SRAM is NOT updated yet!
  2. You start DMA TX                     → DMA reads from SRAM
                                            SRAM has the OLD data
  3. DMA sends old/garbage data over SPI  → Jetson decodes garbage
  4. No error flag anywhere               → silent corruption!

Symptom: first 32 bytes correct (within one cache line), rest is garbage.
         This is the tell-tale sign of a cache coherency bug.
```

**Fix A — Flush cache before DMA TX**:
```c
// Force CPU cache → SRAM so DMA sees fresh data
SCB_CleanDCache_by_Addr((uint32_t*)tx_buf, tx_len);
// NOW start DMA
start_dma_tx(tx_buf, tx_len);
```

**Fix B — Invalidate cache after DMA RX**:
```c
// After DMA fills rx_buf, CPU cache may still have old data
// Invalidate: force CPU to re-read from SRAM on next access
start_dma_rx(rx_buf, rx_len);
// ... wait for TC ISR ...
SCB_InvalidateDCache_by_Addr((uint32_t*)rx_buf, rx_len);
// NOW read rx_buf in CPU code
```

**Fix C (best) — Non-cacheable SRAM section**:
```c
// Put DMA buffers in a region the MPU marks as non-cacheable
// Linker places them in a dedicated section
__attribute__((section(".noncacheable")))
__attribute__((aligned(32)))          // must align to cache line size
static uint8_t tx_buf[256];

// No cache flush/invalidate ever needed for this buffer
// CPU reads/writes go directly to SRAM, DMA always sees correct data
```

---

## FIFO + Burst Mode

Normal (direct) mode: each peripheral request → one bus transaction.
FIFO mode: DMA collects bytes internally, then does one burst → fewer bus transactions.

```
Direct mode (single beat):
  ADC ready → DMA: one 16-bit read → one 16-bit write to SRAM
  100,000 bus transactions/sec for 100kHz ADC

FIFO + burst mode:
  DMA accumulates 4 samples in 16-byte FIFO
  → one 64-bit burst write to SRAM
  25,000 bus transactions/sec (4× reduction)
  → less bus contention → other DMA channels get more bandwidth
```

```c
// Enable FIFO and burst
DMA1_Stream0->FCR |= DMA_SxFCR_DMDIS;    // disable direct mode (= enable FIFO)
DMA1_Stream0->FCR |= DMA_SxFCR_FTH_1;   // fire when FIFO 1/2 full
DMA1_Stream0->CR  |= DMA_SxCR_MBURST_0; // memory burst: 4 beats
DMA1_Stream0->CR  |= DMA_SxCR_MSIZE_1;  // memory width: 16-bit
```

Use FIFO+burst for: ADC, audio (I2S), camera, USB — sustained high throughput.
Keep direct mode for: SPI/UART bytes — irregular timing, small packets.

---

## Real-World Use Cases

### 1. SPI Display — 240×240 LCD (115,200 bytes/frame)
```c
// Without DMA: CPU spends 200ms/frame pushing pixels at 10MHz → nothing else runs
// With DMA: arm transfer, CPU updates game/UI logic, ISR flips framebuffer
HAL_SPI_Transmit_DMA(&hspi1, framebuffer, 240*240*2);
// CPU free immediately
```

### 2. GPS UART — Circular DMA (never stops)
```c
// NMEA sentences at 115200 baud — ~11,500 interrupts/sec without DMA
// With circular DMA: zero interrupts while receiving, CPU parses at leisure
HAL_UART_Receive_DMA(&huart1, gps_ring_buf, sizeof(gps_ring_buf));
// Parser thread scans ring buffer for complete sentences
```

### 3. Microphone — I2S Double Buffer (real-time audio)
```c
// 16kHz, 16-bit = 32,000 bytes/sec forever
// Half-transfer ISR: process first half while DMA fills second half
HAL_I2S_Receive_DMA(&hi2s2, audio_buf, AUDIO_BUF_SAMPLES * 2);
// Wake-word detector processes 32ms chunks with zero gaps
```

### 4. Motor Control ADC — 100kHz @ 3 phases
```c
// 300,000 samples/sec total, must complete current loop in <10µs
// ADC scan mode + DMA circular: samples land in buffer, ISR runs FOC algorithm
HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buf, 3);
// Each ISR: Clarke/Park transform → PI controller → PWM output
```

### 5. Camera Frame Crop — Memory-to-Memory
```c
// Copy center 160×120 region from 320×240 raw frame
// DMA handles 57,600 bytes while CPU sets up next frame processing
for (int row = 60; row < 180; row++) {
    uint8_t *src = full_frame + (row * 320 + 80) * 3;
    uint8_t *dst = crop_buf + ((row-60) * 160) * 3;
    HAL_DMA_Start_IT(&hdma_m2m, (uint32_t)src, (uint32_t)dst, 480);
}
```

### 6. The Cache Bug in Production — STM32H7 Ethernet
```c
// Real bug: engineer prepares Ethernet TX buffer, CPU writes to cache
// DMA reads SRAM (not cache) — sends old packet
// Symptom: first 32 bytes correct, rest garbage — exactly one cache line
// Fix: SCB_CleanDCache_by_Addr() before transmit
// Time to find: 3 days (no error flags anywhere)
```

---

## Summary Table

| Mode | Use Case | Key Property |
|---|---|---|
| Normal M→P | SPI frame TX, UART TX | Stops after N bytes, fires TC ISR |
| Normal P→M | SPI frame RX, I2C RX | Same, fills your buffer |
| Circular | ADC, audio, GPS UART RX | Wraps forever, HT+TC ISR |
| Double buffer (DBM) | High-speed RX, video | HW alternates M0AR/M1AR — no SW swap |
| FIFO + burst | ADC at 100kHz+, camera | Batches bus transactions for throughput |
| Memory → Memory | Fast memcpy, crop | No peripheral clock gating |

**The one-line summary**: DMA is a hired helper that moves data between memory and peripherals while the CPU thinks about something else — essential for anything faster than ~10kHz.
