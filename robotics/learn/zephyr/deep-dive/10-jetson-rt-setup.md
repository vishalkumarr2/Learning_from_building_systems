# 10 — Jetson Orin NX: RT Setup + spidev Baseline
### Jetson Orin NX 8GB · JetPack 5.x (PREEMPT_RT) · spidev · cyclictest

**Status:** 🟡 HARDWARE-GATED  
**Prerequisite:** Session `09` milestone complete; STM32 transmitting nanopb frames over SPI  
**Hardware required:** Jetson Orin NX fully booted, SPI wiring to STM32 complete  
**Unlocks:** `05-100hz-spi-bridge.md` Jetson consumer section, then `06-ros2-publisher.md`  
**Time budget:** ~6 hours  
**Mastery plan:** Project 9 prerequisites + "RT setup before Project 10"

---

## Goal of This Session

Set the Jetson Orin NX for real-time SPI workloads: max performance mode, RT tuning, spidev pinmux, hardware baseline. Verify the raw SPI link at full speed with `spidev_test` before any Python or ROS 2 code.

**Milestone**: `spidev_test -D /dev/spidev0.0 -s 10000000 -p "AABBCCDD" -v` shows loopback data received correctly at 10 MHz. `cyclictest` shows max latency <100µs on isolated CPU core.

---

## Why This Comes Before Python SPI Code

Every beginner writes Python SPI code on the Jetson before verifying the hardware link. When `xfer2()` returns wrong data, they debug Python — but the root cause is:
1. SPI pinmux not configured (no `/dev/spidev0.0` device)
2. `max_speed_hz` left at 500kHz (20× too slow for 100Hz × 128-byte frame = 112kbps needed)
3. GPIO interrupt on Jetson sees SPI CS and locks up without proper IRQ balancing

Do the hardware verification in this session. Only then write Python.

---

## Step 1: Maximum Performance Mode

```bash
# Set NVPModel to maximum performance (no power capping)
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify
sudo nvpmodel -q                 # should show MODE_15W_2CORE or MAXN depending on model
sudo jetson_clocks --show        # all CPU/GPU clocks at maximum

# Make persistent across reboots
sudo systemctl enable nvpmodel
# nvpmodel sets -m 0 at boot when enabled with the right default in /etc/nvpmodel.conf
```

**Why this matters**: Without `jetson_clocks`, CPU cores can throttle to 400MHz during SPI reads. At 10MHz SPI with 128-byte frames at 100Hz, the Jetson CPU has 90ms of spare time per second — but throttled, `ioctl()` latency can spike to 5–20ms, causing missed frames. With max clocks: `ioctl()` consistently <500µs.

---

## Step 2: Stop IRQ Balancing

```bash
# irqbalance migrates IRQs between CPU cores for thermal balance.
# At 100Hz, the SPI IRQ fires 100 times/second.
# Each migration adds ~200µs latency spike and can cause a dropped frame.
sudo systemctl disable irqbalance
sudo systemctl stop irqbalance

# Verify it's stopped
systemctl status irqbalance   # should show "inactive (dead)"
```

---

## Step 3: Isolate a CPU Core for SPI

This prevents the kernel scheduler from running other processes on CPU core 3, reducing SPI ioctl jitter.

```bash
# Add to /boot/extlinux/extlinux.conf — append to APPEND line:
# isolcpus=3 nohz_full=3 rcu_nocbs=3

# Edit the file:
sudo nano /boot/extlinux/extlinux.conf
# Find the APPEND line and add at the end:
#   isolcpus=3 nohz_full=3 rcu_nocbs=3

# Example final APPEND line:
# APPEND ${cbootargs} root=/dev/mmcblk0p1 rw rootwait ... isolcpus=3 nohz_full=3 rcu_nocbs=3

sudo reboot

# After reboot, verify:
cat /sys/devices/system/cpu/isolated   # should show "3"
```

---

## Step 4: Measure RT Latency Baseline

```bash
# Install cyclictest
sudo apt-get install -y rt-tests

# Measure on isolated core 3 for 30 seconds
sudo taskset -c 3 cyclictest \
    --mlockall \
    --priority=99 \
    --interval=1000 \
    --duration=30 \
    --histofall \
    --quiet \
    | tail -20

# Target: max latency < 100µs (typical: <30µs on Jetson Orin with PREEMPT_RT)
# If max latency >200µs: check irqbalance is stopped, check isolcpus applied
```

**Record this number.** You will compare it later when debugging dropped SPI frames. If `cyclictest` shows max 20µs and you still drop frames, the problem is in your application code, not the kernel.

---

## Step 5: Configure SPI Pinmux

**Critical**: `/dev/spidev0.0` does not exist until you configure the SPI pinmux via `jetson-io.py`.

```bash
# Identify which SPI bus connects to STM32 (typically SPI1 on 40-pin header)
# Jetson Orin NX 40-pin header SPI1:
#   Pin 19: MOSI (SPI1_MOSI)
#   Pin 21: MISO (SPI1_MISO)
#   Pin 23: SCLK (SPI1_CLK)
#   Pin 24: CS0  (SPI1_CS0)

# Configure pinmux
sudo /opt/nvidia/jetson-io/jetson-io.py
# In the TUI:
#   Configure Jetson 40pin Header → SPI1 → Enable → Save
#   This writes a device tree overlay and requires reboot

sudo reboot

# After reboot:
ls -la /dev/spidev*
# Should show: /dev/spidev0.0  /dev/spidev0.1  (or similar)

# If no spidev devices: check jetson-io config was saved correctly
# Check dmesg: sudo dmesg | grep spi
```

---

## Step 6: Hardware Loopback Test

**Test with MISO shorted to MOSI (loopback jumper)** before connecting to STM32.

```bash
# Temporarily short Pin 19 (MOSI) to Pin 21 (MISO) on the 40-pin header

# Run spidev_test at 1MHz first
sudo spidev_test -D /dev/spidev0.0 -s 1000000 -p "AABBCCDD01020304" -v
# Expected output: RX bytes match TX bytes exactly

# Increase to 10MHz
sudo spidev_test -D /dev/spidev0.0 -s 10000000 -p "AABBCCDD01020304" -v
# Expected: same match

# Try 16MHz to find the limit
sudo spidev_test -D /dev/spidev0.0 -s 16000000 -p "AABBCCDD01020304" -v
# If this fails: wiring is too long. Use <6 inch wires for 16MHz.

# Remove loopback jumper when done
```

**Failure modes during loopback test:**
| Symptom | Cause |
|---------|-------|
| `spidev_test: can't open device /dev/spidev0.0` | Pinmux not configured; rerun `jetson-io.py` |
| RX bytes all 0x00 | MISO not shorted to MOSI (loopback jumper missing) |
| RX bytes garbage at high speed | Wire too long; reduce to 16cm max for 10MHz |
| `ioctl error -1` | Wrong permissions; use `sudo` or add user to `spidev` group |

---

## Step 7: Connect STM32 and Verify Real Frame

After loopback passes, connect STM32:

```
Jetson 40-pin Header          STM32 Nucleo-H743ZI2
────────────────────          ────────────────────
Pin 19 (MOSI / SPI1_MOSI) ── PA7 (SPI1_MOSI)
Pin 21 (MISO / SPI1_MISO) ── PA6 (SPI1_MISO)
Pin 23 (CLK  / SPI1_CLK)  ── PA5 (SPI1_SCK)
Pin 24 (CS0  / SPI1_CS0)  ── PA4 (SPI1_NSS)
Pin 6  (GND)              ── GND

Logic levels: both 3.3V — direct connection, no level shifting needed.
Wire length: <15cm. SMA or similar for 10MHz+.
```

```bash
# With STM32 armed and transmitting frames (from session 05/09):
sudo python3 -c "
import spidev, time
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 10_000_000
spi.mode = 0   # CPOL=0, CPHA=0 — match STM32 config

for i in range(5):
    rx = spi.xfer2([0]*130)
    length = (rx[0] << 8) | rx[1]
    print(f'Frame {i}: len={length} first_bytes={rx[2:6]}')
    time.sleep(0.01)
spi.close()
"
# Expected: len=47 (or similar), non-zero bytes in payload
```

---

## ioctl Timing Measurement

```python
import spidev, time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 10_000_000
spi.mode = 0

latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    spi.xfer2([0]*130)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1e6)  # microseconds

latencies.sort()
print(f"ioctl latency: mean={sum(latencies)/len(latencies):.0f}µs  "
      f"p50={latencies[500]:.0f}µs  "
      f"p99={latencies[990]:.0f}µs  "
      f"max={latencies[-1]:.0f}µs")
spi.close()

# Target on isolated core with jetson_clocks:
#   mean <500µs, p99 <2ms, max <5ms
# If max >10ms: irqbalance not stopped, or core not isolated
```

---

## Thermal Throttle Guard

```bash
# Install jetson_stats for monitoring
sudo pip3 install jetson-stats

# In a separate terminal during SPI testing:
jtop   # watch CPU temp column; if >65°C, throttling may occur

# Check throttle state:
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
# Compare to max: cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq
# If cur < max, throttling is active → improve cooling
```

---

## Milestone Checklist

- [ ] `nvpmodel -q` shows MAX performance mode
- [ ] `irqbalance` service is stopped and disabled
- [ ] `/sys/devices/system/cpu/isolated` shows `3` after reboot
- [ ] `cyclictest` max latency < 100µs on core 3
- [ ] `/dev/spidev0.0` exists
- [ ] Loopback test passes at 10MHz (`spidev_test` RX == TX)
- [ ] Real STM32 frame: `len=XX` non-zero, non-garbage received
- [ ] ioctl timing: p99 <2ms over 1000 samples
- [ ] Jetson CPU temp <65°C during testing

---

## Pre-Read for Session 11 (EKF)

Before `11-ekf-integration.md`:
1. `ros2 pkg info robot_localization` — understand what it does (EKF fusion node)
2. TF2 frame tree: `map → odom → base_link → imu_link` — all 4 frames must exist
3. `ros2 topic echo /imu/raw` — confirm your session 06 publisher is live
4. `00-mastery-plan.md` Project 11 failure points: zero covariance diagonal, timestamp lag, rviz2 fixed frame

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### RT Setup
- nvpmodel mode: ___
- irqbalance stopped: yes/no
- isolcpus applied (cat /sys/devices/system/cpu/isolated): ___
- cyclictest max latency (30s): ___µs

### SPI Hardware
- /dev/spidev0.0 found: yes/no
- Loopback test 10MHz: pass/fail
- Real STM32 frame len: ___ bytes

### ioctl Timing (1000 samples)
- mean: ___µs  p99: ___µs  max: ___µs

### Issues
- ...
```
