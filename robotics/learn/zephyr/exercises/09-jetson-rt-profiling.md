# Exercises: Jetson Orin NX — RT Setup + SPI Baseline
### Covers: Deep-dive session 10 — nvpmodel, jetson_clocks, isolcpus, cyclictest, spidev loopback, ioctl timing

---

## Section A — Conceptual Questions

**A1.** After flashing JetPack 5.x, what three commands do you run in order to maximise CPU
performance before benchmarking? Explain what each one does at the hardware level.

<details><summary>Answer</summary>

```bash
sudo nvpmodel -m 0          # 1. Set power mode
sudo jetson_clocks          # 2. Lock all clocks at max
sudo systemctl disable irqbalance && sudo systemctl stop irqbalance  # 3. Freeze IRQ affinity
```

**1. `nvpmodel -m 0`** — selects the MAX power envelope. Allows all CPU cores to be active at
maximum frequency. Lower modes (1, 2, etc.) cap TDP by reducing core count or frequency, which
adds governor-driven frequency scaling that introduces latency jitter.

**2. `jetson_clocks`** — pins GPU, CPU, EMC (memory bus), and DLA clocks at their maximum
advertised frequency. Without this, the cpufreq governor (typically `schedutil`) scales clocks
down during idle periods. When your RT thread wakes, it may run at reduced frequency for the first
few microseconds while the PLL ramps up → unpredictable worst-case latency spike.

**3. `systemctl disable irqbalance`** — the `irqbalance` daemon periodically migrates hardware
interrupts between CPU cores to balance load. If it migrates your SPI IRQ away from the isolated
core mid-transfer, the interrupt handler runs on a shared core, adding scheduler contention
latency. Disabling it locks IRQ affinity permanently to the kernel's default assignment.

</details>

---

**A2.** Explain the purpose of `isolcpus=3 nohz_full=3 rcu_nocbs=3` in `/boot/extlinux/extlinux.conf`.
What is "tick noise" and how does `nohz_full` eliminate it? What is an RCU callback and why does
`rcu_nocbs` matter for a 100Hz RT thread?

<details><summary>Answer</summary>

These three kernel command-line parameters together create a **dedicated isolated CPU core** (core 3).

**`isolcpus=3`:** Removes CPU 3 from the Linux scheduler's runqueue. No ordinary (non-RT) processes
will be scheduled there. Only threads that explicitly call `pthread_setaffinity_np()` or
`sched_setaffinity()` to pin themselves to core 3 will run on it.

**`nohz_full=3`:** Suppresses the periodic scheduler tick (normally 250Hz or HZ=1000) on core 3
when there is exactly one runnable task. The tick is a timer interrupt that wakes the kernel
every 1-4ms regardless of workload. On a 100Hz thread (10ms period), one tick interrupt
per period is ~1-10% additional interrupt latency overhead at worst timing. `nohz_full` eliminates
this → the isolated core runs interrupt-free between hardware events.

**`rcu_nocbs=3`:** Read-Copy-Update (RCU) is the kernel's lock-free synchronisation mechanism.
Periodically, the kernel runs RCU callbacks (deferred memory frees, list updates) on each core.
Without `rcu_nocbs`, these callbacks run on core 3 in your RT thread's context window, adding
unpredictable 10-100μs jitter. `rcu_nocbs=3` offloads all RCU callbacks from core 3 to a
dedicated kthread on another core.

Combined: core 3 has **no scheduler ticks, no RCU callbacks, and no involuntary process migrations**
— it is as close to bare-metal as Linux gets.

</details>

---

**A3.** You run `cyclictest --mlockall -t1 -p99 -i 1000 -n -a 3 --duration=30s` and get:

```
T: 0 ( 1234) P:99 I:1000 C:  30000 Min:    8 Act:   11 Avg:   12 Max:   847
```

Interpret every field. Is this result acceptable for a 100Hz (10ms period) SPI bridge? What
would make it unacceptable?

<details><summary>Answer</summary>

- `T: 0 ( 1234)` — Thread 0, PID 1234.
- `P:99` — Running at SCHED_FIFO priority 99 (highest real-time priority).
- `I:1000` — Timer interval 1000 µs (1ms).
- `C: 30000` — 30,000 measurement cycles completed (30,000 × 1ms = 30s).
- `Min: 8` — Minimum latency observed: 8 µs (best case, core idle, immediate wakeup).
- `Act: 11` — Most recent measurement: 11 µs.
- `Avg: 12` — Average latency: 12 µs.
- `Max: 847` — **Worst-case latency: 847 µs** (nearly 1ms).

**Acceptability for 100Hz SPI bridge (10ms period):**

The 847 µs worst case is within the 10ms period — so the thread always meets its deadline, but only
with 9.15ms of margin. This is **acceptable** for a sensor bridge where the protocol tolerates
sub-millisecond jitter.

It would be **unacceptable** if:
- The period were shorter (e.g. 1ms, 500Hz), where 847µs jitter would cause missed deadlines.
- You observe Max > 1000 µs — this indicates a non-RT interrupt or process stole the core.
- You observe Max growing over time (run for 5+ minutes) — suggests a periodic kernel event
  (RCU grace period, memory compaction) that `rcu_nocbs` didn't fully mitigate.
  
Target for production: Max < 100 µs (good), < 500 µs (acceptable), > 1000 µs (investigate).

</details>

---

**A4.** The Jetson spidev physical setup requires a `jetson-io.py` step before `spidev_test` works.
What does `jetson-io.py` configure, and why can't you just `modprobe spidev` and expect
`/dev/spidev0.0` to appear?

<details><summary>Answer</summary>

`jetson-io.py` configures the **pin multiplexer (pinmux)** — the hardware register that determines
whether each GPIO pad is connected to SPI, I2C, UART, or plain GPIO function. On the Jetson Orin
NX, the SPI controller exists in silicon, but by default the pins that would be routed to SPI
are configured as GPIO or another function.

`modprobe spidev` loads the kernel driver, which creates the `/sys/bus/spi/` device tree. But if
the pinmux has not been set to route the SPI signals to the 40-pin header pads, the SPI controller
is connected to nothing — there are no rising/falling edges on any physical pin.

`jetson-io.py` writes the pinmux configuration to the device tree overlay and saves it
persistently. After reboot, the pins are committed to SPI function, the kernel's device tree
enumerates the SPI controller, `spidev` binds to it, and `/dev/spidev0.0` appears.

**Shortcut diagnostic:** After `jetson-io.py` + reboot, run:
```bash
ls /dev/spidev*
# Should show: /dev/spidev0.0  /dev/spidev0.1
```
If nothing appears, check `dmesg | grep spi` — the driver bound but no device node means the
udev rule is missing or the device tree overlay didn't apply.

</details>

---

**A5.** You measure ioctl latency for `SPI_IOC_MESSAGE(1)` on Jetson Orin NX and get p99 = 4.8ms.
Your target is <2ms. Name two hardware causes and two software causes, and for each, the
diagnostic command or change you would try.

<details><summary>Answer</summary>

**Hardware causes:**

1. **CPU not locked at max frequency.** The cpufreq governor scaled down core 3 after the first
   measurement. The SPI transfer starts slowly while the PLL ramps.
   Diagnostic: `cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq` — should match
   `cpuinfo_max_freq`. Fix: re-run `sudo jetson_clocks` and keep it running as a service.

2. **SPI clock rate too low.** At low SPI clock (1MHz), 130 bytes takes 1.04ms + interrupt
   overhead. Raising to 10MHz drops transfer time to 104µs.
   Diagnostic: Check `spi_ioc_transfer.speed_hz` in your code. Test with `spidev_test -D
   /dev/spidev0.0 -s 10000000 -p "DEADBEEF..."`.

**Software causes:**

3. **Thread not pinned to isolated core.** Without `pthread_setaffinity_np` + `SCHED_FIFO`,
   the ioctl call may migrate to a busy core mid-execution.
   Diagnostic: `ps -eLo pid,psr,cls,pri,cmd | grep your_process` — PSR shows current core.
   Fix: call `set_cpu_affinity(3)` and `set_sched_fifo(90)` before the main loop.

4. **Measurement includes Python/ctypes overhead.** If you're measuring with Python `time.perf_counter()`
   around `fcntl.ioctl()`, Python's global interpreter lock (GIL) adds unpredictable overhead.
   Fix: measure from C using `clock_gettime(CLOCK_MONOTONIC_RAW)` immediately before and after
   the ioctl syscall — inside the same C function, not from Python.

</details>

---

## Section B — Practical / Debug Scenarios

**B1.** `spidev_test -D /dev/spidev0.0 -p "DEADBEEF"` returns data but it is all zeros, not the
loopback echo you expected. The MISO and MOSI pins are confirmed connected with a jumper wire.
What is the most likely cause?

<details><summary>Answer</summary>

**SPI mode mismatch (CPOL/CPHA).** The default `spidev_test` uses SPI mode 0 (CPOL=0, CPHA=0).
If the jumper wire loopback is working correctly (MISO=MOSI), you should get an echo. All-zeros
suggests data is not being sampled at the right clock edge.

Secondary possibility: **the `D/R` receive-enable pin on the SN65HVD230 transceiver** (if the
transceiver is in the signal path — but for a bare loopback with jumper, this doesn't apply).

For pure loopback test (no STM32 connected):
```bash
# Force mode 0, speed 1MHz, explicit loopback
spidev_test -D /dev/spidev0.0 -s 1000000 -p "HELLO" -v
```
If still all zeros: check `dmesg | grep spi` for `spi_imx: unhandled interrupt` or FIFO errors.
If there are no errors: the spidev device was opened but the pinmux is still not applied —
`/dev/spidev0.0` exists but the pins are routing to GPIO, not SPI hardware. Repeat `jetson-io.py`
and reboot.

</details>

---

**B2.** `cyclictest` is running on core 3 with `isolcpus=3` set, but Max latency is still 1200 µs
after 60 seconds. `dmesg` shows no errors. What are three things to check?

<details><summary>Answer</summary>

1. **`irqbalance` is still running.** Even after `systemctl disable`, if it was not stopped for
   the current session: `systemctl stop irqbalance`. Verify: `ps aux | grep irqbalance`.

2. **A non-RT interrupt is routed to core 3.** Check `/proc/interrupts` — look for any IRQ line
   with a non-zero count in the "CPU3" column. Common culprits: SPI1, I2C, ethernet. Migrate
   them away with `echo 7 > /proc/irq/<N>/smp_affinity` (bitmask for cores 0,1,2 only).

3. **`rcu_nocbs` is not applied.** Verify the kernel was actually booted with the parameter:
   `cat /proc/cmdline | grep rcu_nocbs`. If absent, the overlay change in `extlinux.conf` didn't
   take effect — check for a syntax error or that the correct `extlinux.conf` was modified (the
   Jetson has multiple and uses the one specified by the bootloader chainloading).

</details>

---
