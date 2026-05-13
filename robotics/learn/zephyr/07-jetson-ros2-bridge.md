# Jetson Orin — SPI Master + ROS2 Bridge

## Role in the Pipeline

The Jetson is the **SPI master** — it drives the clock and decides exactly when each
100Hz transfer happens. This gives deterministic timing on the ROS2 side regardless
of any jitter in the STM32 sensor reading or packing.

```
ROS2 Timer (100Hz)
       │
       ▼
spidev.xfer2()  ──SPI──►  STM32 sends pre-built frame
       │
       ▼
nanopb decode
       │
       ▼
rclpy publish()  ──►  /imu/raw, /wheel_speed, ...
```

---

## Why Jetson as Master (Not STM32)?

| Option | Pro | Con |
|---|---|---|
| Jetson = master | Deterministic ROS2 publish rate, no handshake needed | Jetson must be running before STM32 is useful |
| STM32 = master | STM32 controls exact sensor timing | Needs GPIO "ready" handshake with Jetson, adds 2-way complexity |

Determinism wins for robotics — ROS2 expects consistent 100Hz, not "whenever the MCU is ready."

---

## Linux SPI Interface: spidev

On Linux, SPI is exposed as `/dev/spidev<bus>.<cs>`:

```bash
# List available SPI devices
ls /dev/spidev*
# /dev/spidev0.0   → bus 0, chip select 0
# /dev/spidev0.1   → bus 0, chip select 1

# Check SPI device tree on Jetson Orin
cat /proc/device-tree/spi@3250000/status
```

Python `spidev` library talks to this device:

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)                    # bus=0, cs=0 → /dev/spidev0.0
spi.max_speed_hz = 10_000_000     # 10 MHz
spi.mode = 0b00                   # CPOL=0 CPHA=0: idle low, sample rising edge
spi.bits_per_word = 8
spi.no_cs = False                 # let spidev manage CS automatically

# Full-duplex transfer: sends tx bytes, receives same number of bytes
rx = spi.xfer2([0x00] * 64)      # send 64 dummy bytes, get 64 back
```

**`xfer2` vs `xfer`**:
- `xfer2`: CS stays asserted for entire transfer ← use this
- `xfer`:  CS pulses between each byte ← almost never what you want

**DMA note**: spidev automatically uses kernel DMA for transfers >32 bytes. You don't see it — it just makes large transfers faster.

---

## Complete ROS2 Node

```python
# ros2_ws/src/stm32_bridge/stm32_bridge/spi_node.py

import spidev
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped

from sensor_frame_pb2 import SensorFrame   # generated from sensor_frame.proto


class Stm32BridgeNode(Node):
    SYNC_BYTE   = 0xAA
    MAX_PAYLOAD = 512

    def __init__(self):
        super().__init__('stm32_bridge')

        # ── SPI setup ─────────────────────────────────────────────────
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 10_000_000
        self.spi.mode = 0

        # ── ROS2 publishers ───────────────────────────────────────────
        # BEST_EFFORT QoS: don't queue old sensor data if subscriber is slow
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.imu_pub   = self.create_publisher(Imu,          '/imu/raw',     sensor_qos)
        self.wheel_pub = self.create_publisher(TwistStamped, '/wheel_speed', sensor_qos)

        # ── Stats ──────────────────────────────────────────────────────
        self.rx_ok    = 0
        self.rx_err   = 0
        self.rx_total = 0

        # ── 100Hz transfer timer ───────────────────────────────────────
        self.create_timer(0.010, self._transfer_cb)

        # ── Diagnostic timer: log stats every 5 seconds ───────────────
        self.create_timer(5.0, self._stats_cb)

        self.get_logger().info('stm32_bridge started — 100Hz SPI transfers')

    # ── Main 100Hz callback ────────────────────────────────────────────
    def _transfer_cb(self):
        self.rx_total += 1
        try:
            payload = self._read_frame()
            if payload is None:
                self.rx_err += 1
                return

            frame = SensorFrame()
            frame.ParseFromString(payload)
            self._publish(frame)
            self.rx_ok += 1

        except Exception as e:
            self.get_logger().error(f'SPI transfer error: {e}')
            self.rx_err += 1

    # ── Frame read with sync + CRC ──────────────────────────────────────
    def _read_frame(self) -> bytes | None:
        # Read 3-byte header: sync + 2-byte length
        header = bytes(self.spi.readbytes(3))

        if header[0] != self.SYNC_BYTE:
            # Lost sync — try to re-lock by consuming bytes until 0xAA
            self.get_logger().warn(f'Sync lost (got 0x{header[0]:02X}) — re-syncing')
            self._resync()
            return None

        length = (header[1] << 8) | header[2]

        if length == 0 or length > self.MAX_PAYLOAD:
            self.get_logger().warn(f'Invalid frame length: {length}')
            return None

        # Read payload + 2-byte CRC
        data   = bytes(self.spi.readbytes(length + 2))
        payload = data[:-2]
        rx_crc  = struct.unpack('>H', data[-2:])[0]

        if not self._crc16_ok(payload, rx_crc):
            self.get_logger().warn('CRC mismatch — discarding frame')
            return None

        return payload

    def _resync(self):
        """Read bytes until 0xAA found (sync recovery)."""
        for _ in range(self.MAX_PAYLOAD):
            b = self.spi.readbytes(1)[0]
            if b == self.SYNC_BYTE:
                return

    # ── Publish decoded frame to ROS2 ───────────────────────────────────
    def _publish(self, frame: SensorFrame):
        now = self.get_clock().now().to_msg()

        if frame.HasField('imu'):
            msg = Imu()
            msg.header.stamp    = now
            msg.header.frame_id = 'imu_link'

            msg.linear_acceleration.x = frame.imu.accel_x
            msg.linear_acceleration.y = frame.imu.accel_y
            msg.linear_acceleration.z = frame.imu.accel_z
            msg.angular_velocity.x    = frame.imu.gyro_x
            msg.angular_velocity.y    = frame.imu.gyro_y
            msg.angular_velocity.z    = frame.imu.gyro_z

            # -1.0 in [0] = "covariance unknown" per ROS2 convention
            msg.linear_acceleration_covariance[0] = -1.0
            msg.angular_velocity_covariance[0]    = -1.0

            self.imu_pub.publish(msg)

        if frame.HasField('wheel'):
            msg = TwistStamped()
            msg.header.stamp    = now
            msg.header.frame_id = 'base_link'
            # Pack wheel speeds into twist (convention depends on your robot)
            msg.twist.linear.x  = (frame.wheel.speed_fl + frame.wheel.speed_fr +
                                    frame.wheel.speed_rl + frame.wheel.speed_rr) / 4.0
            self.wheel_pub.publish(msg)

    # ── CRC16-CCITT ──────────────────────────────────────────────────────
    def _crc16_ok(self, data: bytes, expected: int) -> bool:
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        return (crc & 0xFFFF) == expected

    # ── Periodic stats ───────────────────────────────────────────────────
    def _stats_cb(self):
        rate = self.rx_ok / max(self.rx_total, 1) * 100
        self.get_logger().info(
            f'SPI stats: ok={self.rx_ok} err={self.rx_err} '
            f'total={self.rx_total} success_rate={rate:.1f}%'
        )


def main():
    rclpy.init()
    node = Stm32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.spi.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## ROS2 Package Setup

```
ros2_ws/
└── src/
    └── stm32_bridge/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        └── stm32_bridge/
            ├── __init__.py
            ├── spi_node.py
            └── sensor_frame_pb2.py   ← generated from sensor_frame.proto
```

```xml
<!-- package.xml -->
<package format="3">
  <name>stm32_bridge</name>
  <version>0.1.0</version>
  <description>STM32 SPI bridge for Jetson ROS2</description>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <exec_depend>python3-spidev</exec_depend>
  <exec_depend>python3-protobuf</exec_depend>
</package>
```

```python
# setup.py
from setuptools import setup

setup(
    name='stm32_bridge',
    packages=['stm32_bridge'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'stm32_bridge = stm32_bridge.spi_node:main',
        ],
    },
)
```

Build and run:
```bash
cd ~/ros2_ws
colcon build --packages-select stm32_bridge
source install/setup.bash
ros2 run stm32_bridge stm32_bridge

# Verify output
ros2 topic hz /imu/raw         # should show ~100 Hz
ros2 topic echo /imu/raw       # live data
ros2 topic echo /wheel_speed   # live data
```

---

## QoS Settings Explained

```python
# For sensor data — use BEST_EFFORT
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,   # don't retry dropped packets
    history=HistoryPolicy.KEEP_LAST,
    depth=5,                                      # keep latest 5 messages
)

# For control commands or critical data — use RELIABLE
ctrl_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,       # retry until delivered
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
```

**Why BEST_EFFORT for sensors?**
At 100Hz, if a subscriber falls behind, you want the latest IMU reading — not a queue of 50 stale ones from 500ms ago.
RELIABLE on 100Hz data causes unbounded queue growth if any subscriber is slow.

---

## Timing — How 100Hz Is Maintained

```
ROS2 timer resolution on Linux: ~1ms jitter (not real-time)
Actual publish rate: 100Hz ± 1-2Hz typically

For tighter timing (<0.5ms jitter):
  - Use PREEMPT_RT kernel patch on Jetson
  - Set spi_node process to SCHED_FIFO priority:
    sudo chrt -f 80 ros2 run stm32_bridge stm32_bridge
  - Pin process to isolated CPU core:
    taskset -c 3 ros2 run stm32_bridge stm32_bridge
```

---

## SPI Electrical Notes for Jetson Orin

```
Jetson GPIO voltage:      3.3V
STM32 typical SPI:        3.3V  ← compatible, no level shifter needed
STM32F1 (5V tolerant):    5V    ← needs level shifter to Jetson!

Recommended wiring:
Jetson SPI1_CLK  ──────►  STM32 SPI1_SCK
Jetson SPI1_MOSI ──────►  STM32 SPI1_MOSI
Jetson SPI1_MISO ◄──────  STM32 SPI1_MISO
Jetson SPI1_CS0  ──────►  STM32 SPI1_NSS (+ GPIO interrupt pin)
GND              ──────── GND   ← always connect ground between boards!

Max trace length:  ~30cm without impedance matching
Beyond 30cm:       add series 33Ω on CLK/MOSI at source
```

---

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| All bytes 0xFF | STM32 SPI not initialized or DMA not armed | Scope CS — does STM32 respond? Check DMA setup |
| First 32 bytes OK, rest garbage | Cache coherency bug on STM32H7 | Add `SCB_CleanDCache_by_Addr()` before swap |
| CRC errors >1% | Electrical noise or wrong CPOL/CPHA mode | Scope CLK + MOSI, verify `spi.mode` matches STM32 config |
| Rate drops to 50Hz | SPI transfer taking >20ms | Reduce payload size or increase SPI clock |
| Sync byte occasionally wrong | Length field off by 1 | Verify frame builder padding; check `readbytes()` count |
| ROS2 topic empty | spidev permission denied | `sudo usermod -aG spi $USER` or run as root |
