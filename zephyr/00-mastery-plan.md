# Hands-On Mastery Plan: STM32/Zephyr → SPI → Jetson/ROS2 @ 100Hz

> Synthesized from 4 specialist analyses:
> **Embedded firmware gaps** · **ROS2/Jetson gaps** · **Hardware/electronics fundamentals** · **Project curriculum design**

---

## 0. Read This Before Buying Anything

This plan assumes you have read all 17 docs in this folder and now want to actually *build* the system. The docs taught you **what** things are. This plan teaches you **how things break** and in what order to build skills so you don't waste weeks on preventable problems.

**Critical mindset shift:** The logic analyzer is not an advanced tool — it is your primary debug surface. Without it you are debugging blind. Buy it before you write a single line of code.

---

## 1. Hardware Shopping List

### Non-Negotiable (buy first)

| Item | Why | Cost |
|------|-----|------|
| **Saleae Logic 8** (or Logic Pro 8) | Debug everything. SPI/I2C/CAN/UART decode simultaneously. No substitute. | ~$160/$480 |
| **STM32 Nucleo-H743ZI2** or **Nucleo-144** | H7 has D-cache → teaches the real production gotcha. Built-in ST-Link V3. | ~$60 |
| **ICM-42688-P breakout** (SparkFun/Adafruit) | Target IMU. Exact chip matters — register maps differ between chips. | ~$25 |
| **CAN transceiver breakout ×2** (SN65HVD230, 3.3V-compatible) | TWO required: one per CAN node. TJA1050 does NOT work at 3.3V. | ~$10 |
| **120Ω resistors ×4** | CAN bus termination. 2 active + 2 spares. 1/4W. | <$1 |
| **USB-UART adapter** (CP2102 or CH340) | Zephyr shell + log output while ST-Link UART is busy. | ~$8 |
| **GPS UART module** (u-blox M8N or M9N breakout) | Real NMEA parsing. Indoor use = no fix, exactly the edge case you need to handle. | ~$20 |
| **Jumper wire set** (DuPont 20cm, male-female) | Short runs only. SPI at 8MHz dies at 30cm. | ~$10 |
| **Digital multimeter** (any brand) | Verify voltages before connecting anything. | ~$20 |
| **Jetson Orin Developer Kit** | The actual target hardware for the ROS2 side. Raspberry Pi does NOT replicate Tegra SPI timing. | ~$500 |

### Highly Recommended

| Item | Why | Cost |
|------|-----|------|
| Rigol DS1054Z (100MHz scope) | Signal integrity on SPI lines. See ringing, slow edges, CAN dominant/recessive levels. Logic analyzer shows logic, scope shows physics. | ~$350 |
| Bench power supply (30V/3A) | Current-limited power-on. See power draw. Prevents magic smoke. | ~$60 |
| SEGGER J-Link EDU | Enables SystemView thread timeline profiling. Diagnoses priority inversion. | ~$18 |
| USB-CAN adapter (CANable or PCAN-USB) | Linux socketcan/candump. Indispensable for verifying CAN before blaming code. | ~$30 |
| Second Nucleo or Raspberry Pi | SPI master for testing STM32 SPI slave before Jetson is ready. | ~$40 |

---

## 2. Before You Touch Code — Hardware Safety Rules

These are the things that fry hardware. One violation = dead Jetson or dead IMU, and you won't know why.

### Voltage Rule
- Jetson Orin 40-pin header: **3.3V logic**. Measure it with a multimeter before connecting anything.
- STM32 Nucleo: **3.3V** GPIO output.
- ICM-42688: Has TWO power pins — `VDD` and `VDDIO`. Power both from the same 3.3V rail. If `VDDIO` is 1.8V but you drive it with 3.3V STM32 signals → you will damage the IMU.
- SN65HVD230 CAN transceiver: 3.3V VCC compatible. The TJA1050 needs 5V and its logic threshold requires >3.5V input — won't work reliably from a 3.3V STM32.

### Ground Rule
Every device in the system needs a wire to a common GND. USB-powered from two different adapters = two GND references = SPI corruption even when logic analyzer shows "correct" signals.

Minimum GND connections:
- STM32 GND → Jetson GND (40-pin header, pin 6 or 9)
- STM32 GND → IMU GND
- STM32 GND → CAN transceiver GND

### Signal Integrity Rule
SPI at 8MHz with 30cm DuPont wires = ringing + bit errors. Keep SPI wires **under 15cm**. Run a GND wire alongside CLK. Use 0.1" DuPont connectors, not raw breadboard wires for anything over 5cm.

### Pull-up Rule
I2C SDA and SCL **must** have external pull-up resistors to 3.3V. Use **2.2kΩ–4.7kΩ** at 400kHz. Many breakout boards include them — check the breakout schematic. If they're present, don't add more (parallel = lower resistance = overdrive).

### CS Pull-up Rule  
The SPI chip-select (CS/NSS) line **must** be pulled high (to 3.3V via 10kΩ) when not actively driven. It floats during boot, causing accidental transaction starts and latching the IMU into bad states.

### Decoupling Caps Rule
Place a 100nF ceramic capacitor as close as possible (within 2mm) to the VCC pin of every IC. This is not in any software tutorial and will cause random resets and corrupted reads if skipped.

---

## 3. The Day-Wasters — Know These Before They Hit You

These are the specific failure modes that eat multiple days of debugging. The 4 expert agents flagged these repeatedly. Reading about them now will save you.

### STM32-Side Day-Wasters

**D-Cache Coherency (2 days lost if unknown)**  
STM32H7 has D-cache ON by default. DMA writes to SRAM but the CPU reads stale cache. Your SPI TX buffer looks correct in GDB (GDB forces a cache fill), but the Jetson receives garbage. Fix: `SCB_CleanInvalidateDCache_by_Addr(buf, len)` before DMA start, or mark the buffer with `__nocache`.

**SPI Slave DMA Pre-Arming Race (2 days lost)**  
The Jetson asserts CS and starts clocking simultaneously. Your DMA must be armed *before* CS goes low. Arming in the CS interrupt callback fires too late. The first 1–4 bytes of every frame are 0x00. Fix: arm DMA on the *previous transfer complete* callback, not on CS assert.

**nanopb Zero-Field Omission (half a day)**  
Proto3 omits default (zero) fields. `pb_encode` on an all-zero message produces fewer bytes than the maximum. If you size your SPI frame to max and pad the rest, the Jetson tries to decode past the valid boundary when GPS drops fix and all floats go zero. Fix: include an explicit `payload_length` field in the proto and only decode that many bytes.

**Stuck I2C Bus After Power Cycle (1 day)**  
If the IMU was mid-transaction when power died, it holds SDA low. The Zephyr I2C driver returns `-EIO` forever. Fix: manually bit-bang SCL 9 times to clock out the stuck transaction, then assert a STOP condition. Put this recovery in your init code.

**IRQ/Thread Priority Inversion**  
Zephyr IRQ priorities (0 = highest) and thread priorities work backwards from intuition. If your DMA complete IRQ runs at higher HW priority than PendSV (the context switch trigger), the scheduler never runs during DMA bursts. Symptom: one sensor gets all CPU, others starve.

**NVIC + Kconfig Trap**  
Adding `CONFIG_SPI=y` does NOT enable DMA for SPI. You need `CONFIG_SPI_STM32_DMA=y`. The SPI will work without it (CPU polling) but silently fails to meet 100Hz timing budget. No error is thrown.

### Jetson/ROS2-Side Day-Wasters

**Wrong Timestamps (3 days lost in nav stack)**  
Calling `rclcpp::Clock().now()` after the SPI ioctl returns is 1–3ms late. At 100Hz with angular velocity, 2ms = visible EKF drift. Fix: call `clock_gettime(CLOCK_MONOTONIC_RAW, &ts)` *immediately* after ioctl returns. Don't blame IMU calibration or process noise.

**spidev ioctl Blocks Forever on STM32 Reset**  
`ioctl(SPI_IOC_MESSAGE)` has no timeout parameter. If the STM32 resets mid-transfer with CS still asserted, the Jetson's spidev call hangs indefinitely. Your ROS2 node freezes. Fix: run the ioctl in a dedicated thread with `pthread_timedjoin_np` timeout.

**Thermal Throttling Looks Like SPI Timeouts**  
Jetson Orin at sustained load + rosbag + rviz2 hits ~65°C. CPU frequency drops 30%. Your 10ms loop suddenly takes 14ms. You blame SPI. Fix: always run `sudo nvpmodel -m 0 && sudo jetson_clocks` before any timing test, and monitor `tegrastats`.

**irq_balance Migrates Your SPI IRQ**  
`irq_balance` is running by default on JetPack. It migrates your SPI interrupt handler to a different CPU every few minutes, causing 2ms latency spikes that disappear on restart. Fix: `systemctl disable irqbalance`, then manually pin your reader thread with `taskset -c 3`.

**rosbag2 SQLite Stall**  
The default SQLite storage backend flushes its WAL every ~4 seconds, blocking all bag writes for 40ms. During this, DDS backpressure pauses your publisher. Your nav stack sees IMU blackouts. Fix: `--storage mcap` (ROS2 Jazzy+) or use BEST_EFFORT QoS on sensor topics.

**ROS2 Executor Timer Drift**  
`create_wall_timer(10ms, cb)` means "at least 10ms between calls." If any other callback in the same executor takes >2ms, your IMU timer fires at 12ms. At 100Hz that's 83Hz actual. Fix: separate the SPI acquisition into a dedicated RT thread; publish from a queue.

---

## 4. Skills That No Doc Teaches

### STM32/Zephyr

| Gap | What You Learn By Doing |
|-----|------------------------|
| Logic analyzer trigger discipline | Trigger on CS falling edge; add SPI/I2C/CAN decoders; correlate timestamps |
| AF (Alternate Function) numbers | STM32 datasheet pin table — wrong AF = silent GPIO, no error |
| HardFault decode | GDB `info registers` → CFSR, MMFAR + linker map → find the faulting line |
| west flash recovery | `reset_config connect_under_reset`, or BOOT0 to VCC for DFU |
| Zephyr shell as debug surface | `kernel threads`, `kernel stacks`, live variable inspection |
| 100Hz timing budget audit | GPIO toggle + scope measures actual per-function timing |

### Jetson/ROS2/Linux

| Gap | What You Learn By Doing |
|-----|------------------------|
| `spidev_test` baseline | Hardware validation before any application code |
| `cyclictest` + `taskset` | Actual RT latency numbers, fix with `isolcpus` kernel arg |
| `strace -T` on spidev | Real ioctl duration and variance per call |
| `ftrace irqsoff` | Kernel functions holding interrupts off during transfers |
| Pre-allocated ROS2 messages | `malloc` at 100Hz = latency spikes under RT. Mutate + move, don't create |
| Lifecycle nodes | STM32 can be reflashed mid-run; bridge node must survive without killing the robot |
| spi_ioc_transfer struct | `cs_change`, `delay_usecs`, chained transfers for multi-phase transactions |

---

## 5. Project Progression

**One rule: each project must produce a verified, working output before moving to the next.** Do not parallelize. Do not skip the Jetson projects early.

---

### Phase 1 — Zephyr Foundations (STM32 only)

---

#### Project 1: Blinky + Toolchain (~6 hours)

**Goal:** Build and flash a Zephyr app that blinks an LED at a rate set by Kconfig.

**Visible result:** LED blinks. Change `CONFIG_BLINK_MS=250` in `prj.conf`, reflash, blink rate changes.

**New skills:** `west init/build/flash`, `CMakeLists.txt`, `prj.conf`, board overlay, `gpio_pin_toggle_dt()`

**Failure point:** West can't find the ARM toolchain — `ZEPHYR_SDK_INSTALL_DIR` not in `~/.zephyrrc`. Budget 2 hours just for toolchain setup on first use.

---

#### Project 2: UART Shell + Logging (~5 hours)

**Goal:** Add `CONFIG_SHELL` and structured logging; blink rate adjustable at runtime.

**Visible result:** `minicom` shows `[INF] blink: toggling LED`. Type `blink set 500` in shell → rate changes live.

**New skills:** `CONFIG_SHELL`, `CONFIG_LOG`, `LOG_INF/ERR`, `SHELL_CMD_REGISTER`, correct UART instance in overlay

**Failure point:** Two UARTs competing for the same pins — shell and log assigned to the same `uart0` instance produces garbage.

---

#### Project 3: 100Hz Timer Thread + Drift Measurement (~5 hours)

**Goal:** A Zephyr thread fires every 10ms. Log actual `dt_ms` using `k_uptime_get()`.

**Visible result:** Terminal prints `tick=1234 dt_ms=10`. Deliberately overload the thread, watch `dt_ms` drift — understand jitter.

**New skills:** `K_THREAD_DEFINE`, `k_timer_start`, `k_uptime_get_32()`, thread priority levels, `CONFIG_MAIN_STACK_SIZE`

**Failure point:** Default stack (1KB) is too small for LOG calls inside the thread. Thread silently dies. Always set thread stack ≥ 2KB in debug builds.

---

#### Project 4: I2C IMU Read at 100Hz (~10 hours)

**Goal:** Read accelerometer + gyroscope from the ICM-42688 over I2C inside the 100Hz thread.

**Visible result:** `acc_x=-0.12 acc_y=0.03 acc_z=9.81` streaming. Tilt board → values change.

**New skills:** `i2c` devicetree node + overlay, `i2c_write_read()`, WHO_AM_I verification first, I2C pull-up resistor requirements

**Failure point #1:** Wrong address (AD0 pin state sets 0x68 vs 0x69). Returns `-EIO` with no useful message.  
**Failure point #2:** I2C bus stuck from power cycle (SDA held low). Recovery: bit-bang SCL 9 times. Logic analyzer shows which bit it's stuck on.  
**Verification step:** Write an I2C scan loop first. Confirm the IMU physically ACKs its address before debugging Zephyr driver config.

---

#### Project 5: CAN Encoder Receive (~10 hours)

**Goal:** Receive CAN frames from a second node; parse wheel velocity.

**Visible result:** `wheel_vel=1.23` updating live. Using a USB-CAN adapter's `candump` on the host confirms frames are on the bus.

**New skills:** `can_add_rx_filter()`, `can_recv()`, CAN frame struct, bitrate in devicetree, `CONFIG_CAN`, bus termination

**Failure point:** Bitrate mismatch → silent bus-off. No errors. No frames. Check `candump` on the USB-CAN adapter before touching code. Verify 120Ω across CANH/CANL with multimeter (should read ~60Ω with both terminators in circuit and power off).

---

#### Project 6: UART GPS NMEA Parser (~8 hours)

**Goal:** Receive NMEA sentences from a GPS module; parse `$GNGGA` for lat/lon/alt.

**Visible result:** `lat=35.67 lon=139.65 alt=45.2` at 1Hz. With loopback: inject static NMEA strings, verify parse.

**New skills:** `uart_irq_callback_set()`, ring buffer (`ring_buf_get/put`), UART async API, NMEA sentence boundary detection across fragmented reads

**Failure point:** NMEA sentences arrive split across two UART interrupts. Accumulate bytes until `\n` before parsing — never process a single callback as a complete sentence.

---

### Phase 2 — Full Sensor Chain Integration (STM32)

---

#### Project 7: ZBus + nanopb Encode (~12 hours)

**Goal:** Collect all three sensor streams into a single proto struct; encode with nanopb; verify on ZBus subscriber.

**Visible result:** Terminal shows `encoded 47 bytes` at 100Hz, size always identical. ZBus subscriber in a second thread re-decodes and confirms round-trip.

**New skills:** `.proto` file design, `nanopb_generator.py`, `pb_encode()` + `pb_ostream_from_buffer()`, ZBus `ZBUS_CHAN_DEFINE`

**Failure point #1:** `pb_encode()` returns `false` silently. Check `PB_GET_ERROR(stream)`. Almost always a `string` or `bytes` field missing a `max_size` annotation in `.options`.  
**Failure point #2:** Frame size varies (proto3 omits zero-value fields). Design fix: add a `payload_length` field or a length-prefixed framing wrapper. Decide this NOW before the SPI layer is built.  
**Failure point #3:** ZBus drops at 100Hz if the packing thread is too slow. Enable `CONFIG_ZBUS_MSG_SUBSCRIBER_NET_BUF_DYNAMIC` and set packing thread priority high.

---

#### Project 8: SPI Slave DMA at 100Hz (~16 hours — hardest project)

**Goal:** Transfer the nanopb buffer over SPI slave at 100Hz using DMA double-buffering. Verify with logic analyzer.

**Visible result:** Logic analyzer shows 100 clean SPI frames per second. A Python spidev script on a second machine decodes the protobuf correctly.

**New skills:** SPI slave devicetree config, `spi_transceive_async()`, DMA double-buffering, CS GPIO interrupt, `CONFIG_SPI_STM32_DMA`, `CONFIG_SPI_SLAVE`

**Failure point #1 (the big one):** DMA must be armed **before** CS asserts. If you arm it in the CS assert interrupt, you're too late. Arm DMA in the *previous transfer complete* callback.  
**Failure point #2:** STM32H7 D-cache. DMA writes to SRAM, CPU reads stale cache. Call `SCB_CleanInvalidateDCache_by_Addr(buf, len)` before starting DMA. GDB will show correct values (it forces a cache flush) while Jetson receives garbage.  
**Failure point #3:** Buffer alignment. nanopb writes to a stack buffer; DMA on H7 requires 32-byte-aligned buffers in specific memory regions. Use `static uint8_t __aligned(32) tx_buf[MAX_FRAME_LEN]`.

**Budget extra time here. This is where most learners stall for 1–2 weeks.**

---

### Phase 3 — ROS2 and Jetson Integration

---

#### Project 9: Jetson spidev + nanopb Decode (Python) (~8 hours)

**Goal:** Python script on Jetson reads SPI at 100Hz, decodes nanopb, prints sensor fields.

**Visible result:** Jetson terminal shows `imu.acc_x=-0.12 odom.vel=1.23` at 100Hz. Tilt STM32 → Jetson values change within one frame.

**New skills:** `spidev` Python API (`xfer2`, `max_speed_hz`, `mode`), Jetson SPI pinmux via `jetson-io.py` + reboot, protobuf Python (`_pb2.py`), tight loop timing with `time.perf_counter`

**Failure point #1:** `/dev/spidev0.0` does not exist until you run `sudo /opt/nvidia/jetson-io/jetson-io.py` and configure the SPI pinmux, then reboot.  
**Failure point #2:** Default `max_speed_hz` is 500kHz — 20× too slow for 100Hz × frame_size. Set to 8,000,000 explicitly.  
**Before Python SPI code:** run `sudo spidev_test -D /dev/spidev0.0 -s 10000000 -p "AABBCCDD" -v` to verify the raw hardware link works at your target speed.

---

#### Project 10: ROS2 Python Publisher Node (~10 hours)

**Goal:** Wrap the spidev reader into an `rclpy` node publishing `sensor_msgs/Imu`, `nav_msgs/Odometry`, `sensor_msgs/NavSatFix` at 100Hz.

**Visible result:** `ros2 topic hz /imu` → **~100Hz**. In rviz2, IMU axes arrow rotates when you tilt the STM32.

**New skills:** `rclpy.Node`, `create_publisher`, `create_timer`, `SensorDataQoS`, `colcon build` layout, `header.stamp` from clock

**Failure point #1:** Putting the blocking `xfer2()` inside the 100Hz timer callback stalls the executor. Every other callback queues. Fix: spidev reads in a separate Python thread; push to a `queue.Queue`; publish from the timer callback.  
**Failure point #2:** Wrong QoS. Using `RELIABLE` at 100Hz causes backpressure when any subscriber is slow. Use `SensorDataQoS` (BEST_EFFORT, depth=5) for all sensor topics.  
**Failure point #3:** Wrong `frame_id` in the message header (`"imu"` vs `"imu_link"`) — rviz2 shows nothing with no error.

**RT setup before this project:**
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks   # max performance
sudo systemctl disable irqbalance          # stop IRQ migration
# Add to /boot/extlinux/extlinux.conf: isolcpus=3 nohz_full=3 rcu_nocbs=3
```

---

#### Project 11: EKF Integration + Pipeline Validation (~12 hours)

**Goal:** Feed IMU + Odometry into `robot_localization` EKF. Verify fused state in rviz2.

**Visible result:** rviz2 shows `odometry/filtered` updating smoothly at 100Hz. `ros2 run tf2_tools view_frames` shows complete `map→odom→base_link` tree.

**New skills:** `robot_localization` YAML config, TF2 frame tree, `ros2 launch` files, covariance matrix tuning, `rqt_graph` for pipeline visualization

**Failure point #1 (silent data loss):** EKF ignores sensors with `0.0` on the covariance diagonal — zero covariance means "infinitely confident" and triggers NaN propagation. Set every diagonal to a small nonzero value (e.g., `0.01`).  
**Failure point #2 (phantom EKF drift):** Timestamps from `get_clock().now()` called *after* ioctl = 1–3ms late per sample. At 100Hz with angular velocity, this causes EKF velocity drift. Fix: capture `CLOCK_MONOTONIC_RAW` immediately after ioctl completes.  
**Failure point #3 (rviz2 shows nothing):** `Fixed Frame` set to `"map"` but no `map→base_link` TF exists yet. Always start with `Fixed Frame = base_link`.

---

## 6. Time & Effort Estimates

| # | Project | Phase | Est. Hours | Cumulative |
|---|---------|-------|-----------|------------|
| 1 | Blinky + Toolchain | STM32 foundations | 5–7 | 5–7 |
| 2 | Shell + Logging | STM32 foundations | 4–5 | 9–12 |
| 3 | 100Hz Timer | STM32 foundations | 4–6 | 13–18 |
| 4 | I2C IMU | STM32 foundations | 8–12 | 21–30 |
| 5 | CAN Encoder | STM32 integration | 7–10 | 28–40 |
| 6 | UART GPS | STM32 integration | 7–9 | 35–49 |
| 7 | ZBus + nanopb | STM32 integration | 9–12 | 44–61 |
| 8 | SPI Slave DMA | **Critical bridge** | 12–18 | 56–79 |
| 9 | Jetson spidev | Jetson foundations | 6–9 | 62–88 |
| 10 | ROS2 Node | Jetson integration | 8–12 | 70–100 |
| 11 | EKF + Validation | Full pipeline | 10–14 | 80–114 |

**Total: 80–114 hours.**

At 6–8 hours/day focused work: **2–3 weeks full-time**, or 8–12 weeks part-time.

The two projects where learners stall longest: **Project 8** (SPI slave DMA timing) and **Project 11** (TF tree + covariance). Budget double time for both.

---

## 7. Tools to Master (Commands)

### STM32 / Zephyr Shell
```bash
west build -b nucleo_h743zi2 .
west flash
west debug
# Inside Zephyr shell (minicom -b 115200 /dev/ttyACM0):
kernel threads          # see all threads + state
kernel stacks           # check stack usage (detect overflow)
device list             # verify all peripherals initialized
```

### Logic Analyzer (Saleae Logic 2)
```
Trigger: CS falling edge
Decoders: SPI (CPOL=0, CPHA=0, CS active-low), I2C, CAN, UART
Channels: CH0=CLK, CH1=MOSI, CH2=MISO, CH3=CS
Verify: decoded bytes match expected values; no gaps in CS timing
```

### Jetson RT Setup
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks      # max clocks
sudo systemctl disable irqbalance             # pin IRQs to cores
sudo cyclictest -m -sp99 -t4 -l100000         # measure RT latency baseline
chrt -f 90 taskset -c 3 ./spi_reader          # SCHED_FIFO + CPU pin
```

### Jetson Diagnostics
```bash
sudo strace -T -e trace=ioctl ./spi_bridge     # ioctl wall time
tegrastats --interval 100                      # thermal + CPU freq
cat /proc/interrupts | grep spi               # which CPU handles SPI IRQ
sudo spidev_test -D /dev/spidev0.0 -s 10000000 # hardware validation
```

### ROS2 Health Checks
```bash
ros2 topic hz /imu                            # actual publish rate
ros2 topic delay /imu                         # end-to-end latency
ros2 topic info -v /imu                       # full QoS for all connections
ros2 run tf2_tools view_frames && evince frames.pdf   # TF tree
```

---

## 8. Parallel Tracks (Do These in Parallel While Hardware Ships)

While waiting for hardware to arrive:

**Track A — nanopb on host Linux**  
Define your `.proto` schema. Run `nanopb_generator.py`. Write a pure-C program on Linux that encodes and decodes a complete message. Verify round-trip. Find the worst-case `stream.bytes_written`. Add a `payload_length` field now. This is Project 7 minus the STM32.

**Track B — ROS2 on host Linux**  
Install ROS2 Humble/Iron. Write a publisher node that publishes `sensor_msgs/Imu` with a static value. Verify with `ros2 topic echo`. Install `robot_localization`. Read its config format. Build the TF tree on paper. None of this needs Jetson hardware.

**Track C — Read These Specifically**  
- STM32H7 reference manual: Chapter on DMA (MDMA, DMA2), cache-coherent memory regions, DMA buffer alignment requirements  
- ICM-42688-P datasheet: Register map chapter (not just WHO_AM_I — read the FIFO Config and Power Management registers)  
- Linux kernel docs: `Documentation/spi/spidev.rst` (the complete `spi_ioc_transfer` struct reference)  
- ROS2 QoS docs: Reliability + Durability combinations and what happens at subscriber-publisher QoS mismatch

---

## 9. The 5 Questions That Prevent False Debugging 

Before spending more than 30 minutes on a bug, answer these:

1. **"Is the hardware actually doing what I think?"** → Probe it. Logic analyzer first, scope if that's inconclusive.
2. **"Are all GNDs connected?"** → Measure with multimeter. This wastes an entire day at least once for every learner.
3. **"Which end is broken — sender or receiver?"** → Have the sender output a known pattern (0xAA, 0x55). Verify on the other side before adding any real logic.
4. **"Is cache involved?"** → On STM32H7: always suspect D-cache for any DMA corruption. On Jetson: suspect page faults for RT latency spikes.
5. **"Is the thread priority correct?"** → Both Zephyr and Linux have inverted priority semantics from each other. Draw the priority stack on paper.

---

## 10. How You Know You're Done

**STM32 side complete when:**
- Logic analyzer shows 100 clean SPI frames/second with no gaps or corrupt bytes
- Frame rate deviation < 0.5% over 10,000 consecutive frames
- IMU, CAN encoder, and GPS data all present in decoded protobuf
- Intentionally disconnect any sensor mid-run: firmware recovers without hanging
- Stack usage under all-sensors-live is < 80% (verified with `kernel stacks`)

**Jetson side complete when:**
- `ros2 topic hz /imu` shows 100.0 ± 0.5 Hz
- `ros2 topic delay /imu` shows < 5ms end-to-end latency
- EKF fuses IMU + odometry with no NaN, no divergence under 60 seconds of motion
- Adding `ros2 bag record` does not degrade the IMU publish rate
- `cyclictest` latency < 80µs worst case with bridge running

---

*Last updated from 4-agent analysis: embedded firmware, ROS2/Jetson, hardware electronics, and curriculum design perspectives.*
